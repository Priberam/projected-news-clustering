using System;
using System.Collections.Generic;
using System.Linq;
using System.IO;
using MathNet.Numerics.LinearAlgebra;

namespace DistilMonoClustering
{
  public struct DocumentRepresentation
  {
    public DocumentPayload payload;
    public DenseVector representation;
  }

  public class DocumentPayload
  {
    public string id;
    public string group_id;
    public string source_feed_name;
    public string language;
    public Dictionary<string, string> text;
    public DateTime timestamp;
    public DateTime update_timestamp;
    public string forced_label = "";
    public bool allways_merges_to_cluster = false;
    public bool do_not_modify_centroid = false;
    public DenseVector title_repr = new DenseVector();
    public DenseVector paragraph_repr = new DenseVector();
    public DenseVector title_paragraph_repr = new DenseVector();
    public bool use_title_representation = false;
    public bool use_paragraph_representation = false;
    public bool use_title_paragraph_representation = false;
    public bool use_forced_label = false;

    public string UniqueId()
    {
      return id + "_" + group_id;
    }
  }

  public class ClusterDocument
  {
    public string Id { get; set; }
    public Dictionary<string, string> Text { get; set; }
    public Dictionary<string, Object> Translation { get; set; }
    public DateTime Timestamp { get; set; }
    public DateTime UpdateTimestamp { get; set; }
    public String TimestampFormat { get; set; }
    public String Language { get; set; }
    public String GroupId { get; set; }
    public String MediaItemType { get; set; }
    public String CallbackUrl { get; set; }
    public String PageUrl { get; set; }
    public String SourceFeedName { get; set; }
    public DocumentPayload ConvertToDocumentPayload()
    {
      DocumentPayload payload = new DocumentPayload();
      payload.id = Id;
      payload.group_id = "clusters_" + Language;
      payload.language = Language;
      payload.text = Text;
      payload.timestamp = Timestamp;
      payload.update_timestamp = UpdateTimestamp;
      payload.source_feed_name = SourceFeedName;

      return payload;
    }

  }

  public class DatasetSample
  {
    public string id;
    public string text;
    public string title;
    public string event_id;
    public bool duplicate;
    public string lang;
    public string bag_id;
    public DateTime date;
    public string source_feed_name;
    public string cluster;

    public DocumentPayload ConvertToDocumentPayload(bool use_forced_label, bool arbitrary_skip = false)
    {
      DocumentPayload payload = new DocumentPayload();
      payload.id = id;
      payload.group_id = "clusters_" + lang;
      payload.text = new Dictionary<string, string>();
      payload.text["body"] = text;
      payload.text["title"] = title;
      payload.timestamp = date;
      payload.update_timestamp = date;
      payload.source_feed_name = source_feed_name;
      payload.forced_label = cluster;
      payload.language = lang;

      if (use_forced_label && !arbitrary_skip)
        payload.use_forced_label = true;

      return payload;
    }
  }

  public class Cluster
  {
    public Ulid m_record_number;
    public DenseVector m_centroid = new DenseVector();
    public string forced_label;
    public List<DocumentUpdate> m_document_updates = new List<DocumentUpdate>();
    public DateTime m_newest_timestamp = DateTime.MinValue;
    public DateTime m_oldest_timestamp = DateTime.MaxValue;
    public Int32 m_num_documents = 0;
    public Int32 m_complete_num_documents = 0;
    public double m_sum_timestamp = 0;
    public double m_sumsq_timestamp = 0;
    public double m_sumsq_acceptancescores = 0;
    public double m_sum_relevance = 0;
    public int cluster_size_rank_thr = 5;      // set in config
    public double m_sum_similarity = 0;
    public DenseVector title_centroid = new DenseVector();
    public DenseVector paragraph_centroid = new DenseVector();
    public DenseVector title_paragraph_centroid = new DenseVector();
    public DenseVector dist_cluster_centroid = new DenseVector();
    public HashSet<string> available_languages = new HashSet<string>();
    public Dictionary<string, int> label_index = new Dictionary<string, int>();
    public double[] bin_bounds = { 3, 10, 20, 30, 40, 50 };    //used in density function

    public double MeanRelevance()
    {
      return m_sum_relevance / m_num_documents;
    }

    public double RelevanceStamp(double z_score)
    {
      double num = m_num_documents;
      var mean = m_sum_timestamp / num;
      var means_sub = (m_sumsq_timestamp / num) - (mean * mean);
      var std_dev = means_sub >= 0 ? Math.Sqrt(means_sub) : 0;    //ArgumentException NaN was being returned when means_sub < 0
      return mean + (z_score * std_dev);
    }

    public AddPointRes AddClusterFeatures(DenseVector documentRepresentation,
                         DocumentPayload payload,
                         double acceptance_score,
                         double relevance_to_cluster,
                         bool change_representation)
    {

      var span_since_epoch = payload.timestamp.Subtract(new DateTime(1970, 1, 1, 0, 0, 0));
      AddPointRes result = new AddPointRes();

      if (change_representation)
      {
        m_num_documents += 1;

        m_centroid.Add(documentRepresentation);

        m_sum_timestamp += span_since_epoch.TotalHours;

        m_sumsq_timestamp += span_since_epoch.TotalHours * span_since_epoch.TotalHours;

        m_newest_timestamp = new DateTime(Math.Max(payload.timestamp.Ticks, m_newest_timestamp.Ticks));
        m_oldest_timestamp = new DateTime(Math.Min(payload.timestamp.Ticks, m_oldest_timestamp.Ticks));

        m_sumsq_acceptancescores += acceptance_score * acceptance_score;

      }

      m_complete_num_documents += 1;

      if (relevance_to_cluster == -1.0)
      {
        if (m_complete_num_documents < cluster_size_rank_thr)     //check this more carefully, especially the cluster_size_rank_thr setting
        {
          relevance_to_cluster = m_complete_num_documents;
        }
        else
        {
          relevance_to_cluster = (acceptance_score + cluster_size_rank_thr);
        }
      }

      result.relevance = relevance_to_cluster;

      available_languages.Add(payload.language);

      if (payload.id != "")
      {
        DocumentUpdate doc_update = new DocumentUpdate();
        doc_update.document_id = payload.id;
        doc_update.language = payload.language;
        doc_update.group_id = payload.group_id;
        doc_update.forced_label = payload.forced_label;
        doc_update.timestamp = payload.timestamp;
        doc_update.update_timestamp = payload.update_timestamp;
        doc_update.similarity = relevance_to_cluster;

        m_document_updates.Add(doc_update);
        if (!label_index.ContainsKey(payload.forced_label))
          label_index[payload.forced_label] = 0;
        label_index[payload.forced_label] += 1;
      }

      m_sum_relevance += relevance_to_cluster;


      if (payload.use_title_representation && payload.text["title"] != "")
        title_centroid.Add(payload.title_repr);

      if (payload.use_paragraph_representation && payload.paragraph_repr != null)
        paragraph_centroid.Add(payload.paragraph_repr);

      if (payload.use_title_paragraph_representation && payload.title_paragraph_repr != null)
        title_paragraph_centroid.Add(payload.title_paragraph_repr);

      return result;
    }

    public double GetBinFeature()
    {
      double density_val = 0;
      //double[] bin_bounds = { 1, 2, 5, 10, 25, 50, 100, 200, 500 };
      //double[] bin_bounds = { 3, 10, 20, 30, 40, 50 };

      for (var j = 0; j < bin_bounds.Length; j++)
      {
        if (m_num_documents <= bin_bounds[j])
        {
          density_val = 0.1428 * (j + 1);
          break;
        }
        if (m_num_documents > bin_bounds[bin_bounds.Length - 1])
        {
          density_val = 1;
          break;
        }
      }
      return density_val;
    }

    public double GetMeanSimilarity()
    {
      return m_sum_similarity / m_num_documents;
    }
    public void JoinCluster(OuterCluster joiningCluster)
    {
      m_centroid.Add(joiningCluster.m_centroid);
      m_document_updates.AddRange(joiningCluster.m_document_updates);

      foreach (var label in joiningCluster.label_index.Keys)
      {
        if (!label_index.ContainsKey(label))
          label_index[label] = 0;
        label_index[label] += joiningCluster.label_index[label];
      }

      m_newest_timestamp = m_newest_timestamp > joiningCluster.m_newest_timestamp ? m_newest_timestamp : joiningCluster.m_newest_timestamp;
      m_oldest_timestamp = m_oldest_timestamp < joiningCluster.m_oldest_timestamp ? m_oldest_timestamp : joiningCluster.m_oldest_timestamp;
      m_num_documents += joiningCluster.m_num_documents;
      m_complete_num_documents += joiningCluster.m_complete_num_documents;
      m_sum_timestamp += joiningCluster.m_sum_timestamp;
      m_sumsq_timestamp += joiningCluster.m_sumsq_timestamp;
      m_sumsq_acceptancescores += joiningCluster.m_sumsq_acceptancescores;
      m_sum_relevance += joiningCluster.m_sum_relevance;
      title_centroid.Add(joiningCluster.title_centroid);
      paragraph_centroid.Add(joiningCluster.paragraph_centroid);
      title_paragraph_centroid.Add(joiningCluster.title_paragraph_centroid);
      dist_cluster_centroid.Add(joiningCluster.dist_cluster_centroid);
      available_languages.Concat(joiningCluster.available_languages);
    }

    public virtual DenseVector GetMainRepresentation()
    {
      return m_centroid;
    }

  }

  public class OuterCluster : Cluster
  {
    public bool m_is_updated = false;
    public bool m_is_archived = false;

    public OuterCluster()
    {

    }
    public OuterCluster(ClusterRow clusterRow)
    {
      m_record_number = Ulid.Parse(clusterRow.ClusterId);
      m_newest_timestamp = clusterRow.NewestTimestamp;
      m_oldest_timestamp = clusterRow.OldestTimestamp;
      m_num_documents = clusterRow.NumDocuments;
      m_complete_num_documents = clusterRow.CompleteNumDocuments;
      m_sum_timestamp = clusterRow.SumTimestamp;
      m_sumsq_timestamp = clusterRow.SumSqTimestamp;
      m_sumsq_acceptancescores = clusterRow.SumSqAcceptanceScores;
      m_sum_relevance = clusterRow.SumRelevance;
      m_centroid = GetDenseVectorFromBytes(clusterRow.Centroid);
      title_centroid = GetDenseVectorFromBytes(clusterRow.TitleCentroid);
      paragraph_centroid = GetDenseVectorFromBytes(clusterRow.ParagraphCentroid);
      title_paragraph_centroid = GetDenseVectorFromBytes(clusterRow.TitleParagraphCentroid);
    }


    public DenseVector GetDenseVectorFromBytes(Byte[] byteArray)
    {
      DenseVector dense_vector = new DenseVector();

      MemoryStream mem_stream = new MemoryStream(byteArray);
      BinaryReader binary_reader = new BinaryReader(mem_stream);
      var ix = 0;

      for (int i = 0; i < mem_stream.Length; i += 8)
      {
        double el = binary_reader.ReadDouble();
        dense_vector.dense_vector[ix] = el;
        ix += 1;
      }

      binary_reader.Close();
      return dense_vector;
    }

    public AddPointRes AddPoint(DenseVector documentRepresentation,
                         DocumentPayload payload,
                         double acceptance_score,
                         double relevance_to_cluster,
                         bool change_representation)
    {
      AddPointRes result = AddClusterFeatures(documentRepresentation,
                         payload,
                         acceptance_score,
                         relevance_to_cluster,
                         change_representation);

      m_sum_similarity = documentRepresentation.CosineSimilarity(m_centroid);

      return result;

    }


  }

  public struct MicroSimilarity
  {
    public double micro_similarity;
    public int document_count;
    public double cluster_similarity;
  }

  public struct AddPointRes
  {
    public double relevance;
    public string log;
  }

  public class DenseVector
  {
    public Vector<double> dense_vector;
    public double? cached_sq_norm = null;
    public bool refresh_cache = true;

    public DenseVector()
    {
      dense_vector = Vector<double>.Build.Dense(512);
    }

    public DenseVector(double[] vector)
    {
      dense_vector = Vector<double>.Build.Dense(vector);
    }

    public void Add(DenseVector other)
    {
      dense_vector = dense_vector.Add(other.dense_vector);
      refresh_cache = true;
      //dense_vector = dense_vector.Zip(other.dense_vector, (x, y) => x + y).ToArray();
    }

    public void Subtract(DenseVector other)
    {
      dense_vector = dense_vector.Subtract(other.dense_vector);
      refresh_cache = true;
      //dense_vector = dense_vector.Zip(other.dense_vector, (x, y) => x - y).ToArray();
    }

    public double DotProduct(DenseVector other)
    {
      return dense_vector.DotProduct(other.dense_vector);
      //return dense_vector.Zip(other.dense_vector, (x, y) => x * y).Sum();
    }

    public double GetCachedSquaredNorm()
    {
      if (refresh_cache)
      {
        cached_sq_norm = SquaredNorm();
        refresh_cache = false;
      }
      return (double)cached_sq_norm;
    }

    public double SquaredNorm()
    {
      return DotProduct(this);
    }

    public double CosineSimilarity(DenseVector other)
    {
      double norm = GetCachedSquaredNorm();
      double norm_other = other.GetCachedSquaredNorm();

      //double norm = SquaredNorm();
      //double norm_other = other.SquaredNorm();

      if (norm == 0.0 || norm_other == 0.0)
        return 0.0;

      double dp = DotProduct(other);
      return dp / Math.Sqrt(norm * norm_other);
    }

    public SortedDictionary<string, double> FeatureCosineSimilarity(DenseVector other)
    {
      SortedDictionary<string, double> features = new SortedDictionary<string, double>();
      features.Add("embedding_all", this.DotProduct(other));

      return features;
    }
  }

  public class ClassifierModel
  {
    public bool is_loaded = false;
    public FeatureCollection weights = new FeatureCollection();
    public double b = 0.0;
    public bool use_threshold = false;
    public Int32 use_top_ranking = 5;
    public bool is_polynomial_model = false;
    public LibSvmModel polynomial_model = new LibSvmModel();

    double Classify(FeatureCollection instance)
    {
      double membership = this.weights.DotProduct(instance) - b;
      return (1.0 / (1.0 + Math.Exp(-membership)));
    }

    public double Score(FeatureCollection instance)
    {
      if (is_polynomial_model)         //similar approach to the original clustering
        return this.polynomial_model.Score(instance);
      return this.weights.DotProduct(instance) + b;
    }

/*
    public double DotProduct(FeatureCollection instance)
    {
      double value = 0.0;

      foreach (var key in this.weights.Keys)
      {
        double weightVal;
        double instanceVal;
        if (!this.weights.TryGetValue(key, out weightVal))
        {
          weightVal = 0.0;
        }
        if (!instance.TryGetValue(key, out instanceVal))
        {
          instanceVal = 0.0;
        }
        value += weightVal * instanceVal;
      }

      return value;
    }
*/
  }

  public class LibSvmModel
  {
    public double rho;
    public double gamma;
    public double coef0;
    public double degree;
    public List<FeatureCollection> model_sv = new List<FeatureCollection>();

    public double DotProduct(FeatureCollection instance, int line)
    {
      return this.model_sv[line].DotProduct(instance);
    }

/*
    public double DotProduct(FeatureCollection instance, int line)
    {
      double value = 0.0;

      foreach (var key in this.model_sv[line].Keys)
      {
        double weightVal;
        double instanceVal;
        if (!this.model_sv[line].TryGetValue(key, out weightVal))
        {
          weightVal = 0.0;
        }
        if (!instance.TryGetValue(key, out instanceVal))
        {
          instanceVal = 0.0;
        }
        value += weightVal * instanceVal;
      }

      return value;
    }
    */


    public double KernelFunction(FeatureCollection instance, int line)
    {
      double base_pow = gamma * this.DotProduct(instance, line) + coef0;
      return Math.Pow(base_pow, degree);
    }

    public double Score(FeatureCollection instance)
    {
      double sum = 0;

      for (int i = 0; i < model_sv.Count; i++)
      {
        sum += model_sv[i].sv_coef * KernelFunction(instance, i);
      }

      sum -= rho;

      return sum; // sum > 0 ? 1 : -1
    }

  }


  public class DocumentUpdate
  {
    public string document_id;
    public string language;
    public string group_id;
    public string forced_label;
    public double similarity;
    public DateTime timestamp;
    public DateTime update_timestamp;

    public DocumentUpdate()
    {

    }

    public DocumentUpdate(DocumentUpdateRow documentRow)
    {
      document_id = documentRow.DocumentId;
      language = documentRow.Language;
      group_id = documentRow.GroupId;
      forced_label = documentRow.ForcedLabel;
      similarity = documentRow.Similarity;
      timestamp = documentRow.Timestamp;
      update_timestamp = documentRow.UpdateTimestamp;
    }


  };

  public class PersistentState
  {
    public struct OldestCluster
    {      //initialize this   
      public double relevance_stamp;
      public Int32 index;
    }

    public Int64 candidate_i = -1;
    public List<OuterCluster> active_clusters = new List<OuterCluster>();
    public List<OuterCluster> inactive_clusters = new List<OuterCluster>();
    public OldestCluster oldest_cluster;
  }

  public struct ClusterResult
  {
    public double relevance_to_cluster;
    public bool is_decided;
    public Ulid cluster_id;
    public HashSet<string> available_languages;
    public OuterCluster outer_cluster;
  }

  public class BestClusterInfo
  {
    public bool found;
    public int clusterindex;
    public string cluster_forced_label;
    public double rank;
    public FeatureCollection similarity = new FeatureCollection();
  }

  public class ClusterRankResult
  {
    public double rank;
    public FeatureCollection similarity = new FeatureCollection();
    public string cluster_forced_label;
    public List<MicroSimilarity> micro_cluster_distances;
  }

  
  public class FeatureCollection
  {
    public double embedding_all;
    public double newest_ts;
    public double oldest_ts;
    public double relevance_ts;
    public double title_repr;
    public double paragraph_repr;
    public double title_paragraph_repr;
    public double paragraph_centroid;
    public double title_paragraph_centroid;
    public double cluster_density;
    public double mean_similarity;
    public double target_density;
    public double sv_coef; //for libsvm models

    public double[] GetFeatureArray()
    {
      double[] values = new double[] { embedding_all, newest_ts, oldest_ts, relevance_ts, title_repr, paragraph_repr, title_paragraph_repr, paragraph_centroid, title_paragraph_centroid, cluster_density, mean_similarity, target_density};

      return values;
    }

    public string[] GetFeatureStringArray()
    {
      string[] values = new string[] { "embedding_all", "NEWEST_TS", "OLDEST_TS", "RELEVANCE_TS", "title_repr", "paragraph_repr", "title_paragraph_repr", "paragraph_centroid", "title_paragraph_centroid", "cluster_density", "mean_similarity", "target_density"};
      return values;
    }

    public double DotProduct(FeatureCollection features)
    {
      var source_features = this.GetFeatureArray();
      var target_features = features.GetFeatureArray();
      double result = 0;
      for (int i = 0; i < source_features.Length; i += 1)
      {
        result += source_features[i] * target_features[i];
      }
      return result;
    }

    public void ParseFeatureArray(double[] values)
    {
      embedding_all = values[0];
      newest_ts = values[1];
      oldest_ts = values[2];
      relevance_ts = values[3];
      title_repr = values[4];
      paragraph_repr = values[5];
      title_paragraph_repr = values[6];
      paragraph_centroid = values[7];
      title_paragraph_centroid = values[8];
      cluster_density = values[9];
      mean_similarity = values[10];
      target_density = values[11];

    }


  }

  public class ClusterMergeResult
  {
    public double score;
    public OuterCluster cluster;
    public ClusterRankResult rank_result;
  }

  public class PutDocumentResult
  {
    public double relevance_to_cluster;
    public Ulid decision;
    public HashSet<string> available_languages;
    public int best_clusterindex;
    public List<DocumentUpdate> document_updates;
    public bool remove_from_index = false;
  }

  public class ClusteringUpdate
  {
    public string type;
    public string group_id;
    public string cluster_id;
    public string document_id;
    public List<string> top_words = new List<string>();
    public double relevance_to_cluster;
    public HashSet<string> available_languages;
    public int best_clusterindex;
    public List<DocumentUpdate> document_updates;
    public bool remove_from_index = false;
  }

  public struct ClusteringApiUpdate
  {
    public List<ClusteringUpdate> updates;
  }

  public struct NearestClusterElement
  {
    public string cluster_id;
    public double relevance_to_cluster;
  }

  public struct NearestClusterResult
  {
    //using PutDocumentResult since it has a cluster id + relevance to cluster as its attributes
    public string source_cluster;
    public List<NearestClusterElement> target_clusters;
  }

  public struct RemoteClusterId
  {
    public string clusterId;
  }

  public class ClusterIndex
  {
    public Dictionary<Ulid, OuterCluster> clusterIndex = new Dictionary<Ulid, OuterCluster>();
    public Dictionary<string, OuterClusterState> clusterStates = new Dictionary<string, OuterClusterState>();
    public Dictionary<string, DocumentState> documentStates = new Dictionary<string, DocumentState>();
    public void InsertClusterState(OuterClusterState cluster_state)
    {
      var clu_key = cluster_state.cluster.m_record_number.ToString();
      if (cluster_state.state == ClusterDbState.Add)
        clusterStates[clu_key] = cluster_state;

      if (cluster_state.state == ClusterDbState.Update)
      {
        if (clusterStates.Keys.Contains(clu_key))
        {
          clusterStates[clu_key] = new OuterClusterState(clusterStates[clu_key].state, cluster_state.cluster);
        }
        else
          clusterStates[clu_key] = cluster_state;
      }
      if (cluster_state.state == ClusterDbState.Remove)
      {
        if (clusterStates.Keys.Contains(clu_key) && clusterStates[clu_key].state == ClusterDbState.Add)
        {
          clusterStates.Remove(clu_key);
        }
        else
          clusterStates[clu_key] = cluster_state;
      }
    }

    public void InsertDocumentState(DocumentState document_state)
    {
      var doc_key = document_state.document.DocumentId;
      if (document_state.state == ClusterDbState.Add)
        documentStates[doc_key] = document_state;

      if (document_state.state == ClusterDbState.Update)
      {
        if (documentStates.Keys.Contains(doc_key))
        {
          documentStates[doc_key] = new DocumentState(documentStates[doc_key].state, document_state.document);
        }
        else
          documentStates[doc_key] = document_state;
      }

      if (document_state.state == ClusterDbState.Remove)
      {
        if (documentStates.Keys.Contains(doc_key) && documentStates[doc_key].state == ClusterDbState.Add)
        {
          documentStates.Remove(doc_key);
        }
        else
          documentStates[doc_key] = document_state;
      }
    }
  }

  public enum ClusterDbState
  {
    None = 0,
    Add = 1,
    Update = 2,
    Remove = 3
  }

  public class OuterClusterState
  {
    public ClusterDbState state;
    public OuterCluster cluster;

    public OuterClusterState(ClusterDbState state, OuterCluster cluster)
    {
      this.state = state;
      this.cluster = cluster;
    }
  }

  public class DocumentState
  {
    public ClusterDbState state;
    public DocumentUpdateRow document;

    public DocumentState(ClusterDbState state, DocumentUpdateRow document)
    {
      this.state = state;
      this.document = document;
    }


  }



  public class ClusterJoinSample
  {
    public FeatureCollection similarity;
    public int label;
    public ClusterJoinSample(FeatureCollection similarity, int label)
    {
      this.similarity = similarity;
      this.label = label;
    }
  }

  public class EvalResults
  {
    public double tp;
    public double fp;
    public double tn;
    public double fn;
    public double acc;
    public double p;
    public double r;
    public double f1;
    public double n_clu;
  }

}