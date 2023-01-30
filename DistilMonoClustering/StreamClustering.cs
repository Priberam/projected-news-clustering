using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Net.Http;
using MathNet.Numerics.LinearAlgebra;
using System.Text;
using System.Threading.Tasks;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using JsonRazor.Serialization;
using System.Threading;
using Microsoft.EntityFrameworkCore;

namespace DistilMonoClustering
{

  public static class StaticHttpClient
  {
    public const int MaxTimeoutSeconds = 600; // 10min
    private static readonly Lazy<HttpClient> lazy =
           new Lazy<HttpClient>(() =>
           {
             var client = new HttpClient();
             // Configure max timeout, request specific timeout can be configured per request
             // (can only be smaller than this - if needed change this default)
             client.Timeout = TimeSpan.FromSeconds(MaxTimeoutSeconds);
             return client;
           });
    public static HttpClient Instance { get { return lazy.Value; } }
  }
  public class StreamClustering
  {
    CultureInfo inv_c = CultureInfo.InvariantCulture;
    private IConfigurationRoot configuration;
    private ILogger logger;
    double merge_thr;
    double relevance_stamp_zscore;
    double gaussian_std_dev;
    double m_oldest_cluster_relevance_stamp = double.MaxValue;
    int m_oldest_cluster_index = -1;
    ClusterIndex m_cluster_index = new ClusterIndex();
    public PersistentState m_state = new PersistentState();
    ClassifierModel m_classifier = new ClassifierModel();
    ClassifierModel m_merge_classifier = new ClassifierModel();
    ClassifierModel m_clusterjoin_classifier = new ClassifierModel();
    bool config_use_title_representation;
    bool config_use_paragraph_representation;
    bool config_use_title_paragraph_representation;
    bool config_use_cluster_density;
    bool config_use_mean_similarity;
    bool config_use_timestamp_features;
    string config_stored_vectors_path;
    string config_distil_service_url;
    bool config_use_stored_vectors;
    bool config_generate_new_stored_vectors;
    bool config_generate_rank_examples;
    bool config_generate_merge_examples;
    bool config_generate_extra_neg_merge_examples;
    bool config_generate_clusterjoin_examples;
    bool config_enable_save_cluster_pool;
    bool config_evaluate_cluster_recall;
    bool config_random_negative_merge_samples;
    bool config_check_cluster_merge;
    bool config_use_forced_label;
    int qid = 0;
    long stream_position = 0;
    long config_k_first_clusters;
    public string boot_time;
    DateTime latest_insertion_timestamp;
    string module_name;
    private DbContextOptionsBuilder<NearestClusterDbContext> _connectionOptionsBuilder;
    public List<double> toppling_occurrences = new List<double>();
    public Dictionary<string, List<OuterCluster>> label_index = new Dictionary<string, List<OuterCluster>>();
    public int improving_merges = 0;
    int save_cluster_pool_interval;
    public object clusterpool_lock = new object();

    public StreamClustering(IConfigurationRoot configuration, ILogger logger)
    {
      this.configuration = configuration;
      this.logger = logger;
      this.relevance_stamp_zscore = double.Parse(configuration[$"Main:ClusterOptions:RelevanceStampZScore"], inv_c);
      config_k_first_clusters = long.Parse(configuration[$"Main:ClusterOptions:KFirstClusters"], inv_c);
      this.merge_thr = double.Parse(configuration[$"ModelConfig:MergeThr"], inv_c);
      this.gaussian_std_dev = double.Parse(configuration[$"Main:ClusterOptions:DateDaysGaussianStdDev"], inv_c);
      this.boot_time = DateTime.Now.Ticks.ToString();
      this.module_name = configuration[$"Main:ScenarioName"];
      this.save_cluster_pool_interval = int.Parse(configuration[$"Main:ClusterOptions:SaveClusterPoolInterval"], inv_c);
      this.config_enable_save_cluster_pool = bool.Parse(configuration[$"Main:EnableSaveClusterPool"]);
      this.config_distil_service_url = configuration[$"Main:DistilService:Url"];
      this.Load(module_name);
    }
    public void Load(string module_name)
    {
      logger.LogInformation("Loading module " + module_name);
      if (module_name == "zho")
        return;

      LoadLiblinearRank(m_classifier, $"RankingModels:" + module_name);

      if (bool.Parse(configuration[$"LibSvm:" + module_name]))
        LoadLibSvmMerge(module_name);
      else
        LoadLiblinearMerge(m_merge_classifier, $"MergeModels:" + module_name);

      LoadLiblinearMerge(m_clusterjoin_classifier, $"ClusterJoinModels:" + module_name);

      config_use_title_representation = bool.Parse(configuration[$"ModelConfig:UseTitleRepresentation"]);
      config_use_paragraph_representation = bool.Parse(configuration[$"ModelConfig:UseParagraphRepresentation"]);
      config_use_title_paragraph_representation = bool.Parse(configuration[$"ModelConfig:UseTitleParagraphRepresentation"]);
      config_use_cluster_density = bool.Parse(configuration[$"ModelConfig:UseClusterDensity"]);
      config_use_mean_similarity = bool.Parse(configuration[$"ModelConfig:UseMeanSimilarity"]);
      config_use_timestamp_features = bool.Parse(configuration[$"ModelConfig:UseTimestampFeatures"]);
      config_stored_vectors_path = configuration[$"TrainEval:StoredVectorsPath"];
      config_use_stored_vectors = bool.Parse(configuration[$"TrainEval:UseStoredVectors"]);
      config_generate_new_stored_vectors = bool.Parse(configuration[$"TrainEval:GenerateNewStoredVectors"]);
      config_generate_rank_examples = bool.Parse(configuration[$"TrainEval:GenerateRankExamples"]);
      config_generate_merge_examples = bool.Parse(configuration[$"TrainEval:GenerateMergeExamples"]);
      config_generate_extra_neg_merge_examples = bool.Parse(configuration[$"TrainEval:GenerateExtraNegMergeExamples"]);
      config_generate_clusterjoin_examples = bool.Parse(configuration[$"TrainEval:GenerateClusterJoinExamples"]);
      config_evaluate_cluster_recall = bool.Parse(configuration[$"TrainEval:EvaluateClusterRecall"]);
      config_random_negative_merge_samples = bool.Parse(configuration[$"TrainEval:RandomNegativeMergeSamples"]);
      config_check_cluster_merge = bool.Parse(configuration[$"ModelConfig:CheckClusterMerge"]);
      config_use_forced_label = bool.Parse(configuration[$"TrainEval:UseForcedLabel"]);

      if (bool.Parse(configuration[$"Main:EnableSaveClusterPool"]))
      {
        var _connectionString = configuration[$"Main:ConnectionString"];
        _connectionOptionsBuilder = new DbContextOptionsBuilder<NearestClusterDbContext>();
        _connectionOptionsBuilder.UseSqlite(_connectionString);
        var dbcontext = new NearestClusterDbContext(_connectionOptionsBuilder.Options);
        dbcontext.Database.EnsureCreated(); // simple db = simple approach.

        LoadClusterPool();          //load existing clusters from db
        latest_insertion_timestamp = LoadLastDocumentDatetime(module_name);
      }

    }

    public void LoadLiblinearRank(ClassifierModel classifier, string config_string)
    {

      string input_path = configuration[config_string];
      string[] lines = File.ReadAllLines(input_path);
      Dictionary<string, double> weights = new Dictionary<string, double>();

      for (var i = 0; i < lines.Length; i++)
      {
        var segments = lines[i].Split('\t');
        weights[segments[0]] = double.Parse(segments[1], inv_c);
      }

      string[] values = classifier.weights.GetFeatureStringArray();

      double[] features = new double[values.Length];

      for (int i = 0; i < values.Length; i += 1)
      {
        if (weights.Keys.Contains(values[i]))
        {
          features[i] = weights[values[i]];
        }
      }

      classifier.weights.ParseFeatureArray(features);
      classifier.is_loaded = true;

    }

    public void LoadLiblinearMerge(ClassifierModel classifier, string config_string)
    {

      string input_merge_path = configuration[config_string];
      var lines = File.ReadAllLines(input_merge_path);
      Dictionary<string, double> merge_weights = new Dictionary<string, double>();
      classifier.b = double.Parse(lines[0], inv_c);

      for (var i = 1; i < lines.Length; i++)
      {
        var segments = lines[i].Split('\t');
        merge_weights[segments[0]] = double.Parse(segments[1], inv_c);
      }
      string[] values = classifier.weights.GetFeatureStringArray();

      double[] features = new double[values.Length];

      for (int i = 0; i < values.Length; i += 1)
      {
        if (merge_weights.Keys.Contains(values[i]))
        {
          features[i] = merge_weights[values[i]];
        }
      }

      classifier.weights.ParseFeatureArray(features);
      classifier.is_loaded = true;
    }

    public void LoadLibSvmMerge(string module_name)   //called after checking if model is from liblinear or libsvm
    {
      string input_merge_path = configuration[$"MergeModels:" + module_name];
      string[] lines = File.ReadAllLines(input_merge_path);
      m_merge_classifier.polynomial_model.degree = double.Parse(lines[2].Split(' ')[1], inv_c);
      m_merge_classifier.polynomial_model.gamma = double.Parse(lines[3].Split(' ')[1], inv_c);
      m_merge_classifier.polynomial_model.coef0 = double.Parse(lines[4].Split(' ')[1], inv_c);
      m_merge_classifier.polynomial_model.rho = double.Parse(lines[7].Split(' ')[1], inv_c);
      for (var i = 9; i < lines.Length; i++)
      {
        // 1 1:1.190559 2:0.999994 3:0.942779 4:0.994629 
        // sv_coef

        FeatureCollection merge_weights = new FeatureCollection();
        string[] segments = lines[i].Split(' ');
        merge_weights.sv_coef = double.Parse(segments[0], inv_c);

        // configure the connections between feature number and feature name in appsettings.json?
        double[] features_allowed = new double[] { 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0 };  //no title_repr and density features

        double[] features_parsed = new double[features_allowed.Length];

        var fn = 1;
        for (var feat = 0; feat < features_parsed.Length; feat++)
        {
          if (features_allowed[feat] == 1)
            features_parsed[feat] = double.Parse(segments[fn].Split(":")[1], inv_c);
          fn += 1;
        }
        merge_weights.ParseFeatureArray(features_parsed);
        //merge_weights[features_ordered[fn - 1]] = double.Parse(segments[fn].Split(":")[1], inv_c);
        m_merge_classifier.polynomial_model.model_sv.Add(merge_weights);

      }
      m_merge_classifier.is_loaded = true;
      m_merge_classifier.is_polynomial_model = true;
    }

    public void SaveClusterPool()
    {
      lock (clusterpool_lock)
      {
        using (var db = new NearestClusterDbContext(_connectionOptionsBuilder.Options))
        {
          foreach (var clusterState in m_cluster_index.clusterStates.Values)
          {
            if (clusterState.state == ClusterDbState.Add)
            {
              ClusterRow cluster_row = new ClusterRow(clusterState.cluster);
              //Console.WriteLine("Adding cluster " + clusterState.cluster.forced_label);
              db.Add(cluster_row);
            }
            if (clusterState.state == ClusterDbState.Update)
            {
              var existing = db.ClusterRows.Where(d => d.ClusterId == clusterState.cluster.m_record_number.ToString()).FirstOrDefault();

              if (existing != null)
              {
                //Console.WriteLine("Updating cluster " + clusterState.cluster.forced_label);
                existing.UpdateRow(clusterState.cluster);
                db.Update(existing);
              }
              //var db_clu = new ClusterRow(clusterState.cluster);
              //db.ClusterRows.Attach(db_clu);
              //db_clu.UpdateRow(clusterState.cluster);
            }
            if (clusterState.state == ClusterDbState.Remove)
            {
              //Console.WriteLine("Removing cluster " + clusterState.cluster.forced_label);
              var db_clu = new ClusterRow() { ClusterId = clusterState.cluster.m_record_number.ToString() };
              db.Entry(db_clu).State = EntityState.Deleted;
            }
          }

          foreach (var documentState in m_cluster_index.documentStates.Values)
          {
            if (documentState.state == ClusterDbState.Add)
            {
              //Console.WriteLine("Adding doc " + documentState.document.DocumentId);
              db.Add(documentState.document);
            }
            if (documentState.state == ClusterDbState.Update)
            {
              var existing = db.DocumentUpdateRows.Where(d => d.DocumentId == documentState.document.DocumentId).FirstOrDefault();

              if (existing != null)
              {
                //Console.WriteLine("Updating doc " + documentState.document.DocumentId);
                existing.UpdateRow(documentState.document);
                db.Update(existing);
              }
              //var db_doc = documentState.document;
              //db.DocumentUpdateRows.Attach(db_doc);
              //db_doc.UpdateRow(documentState.document);
            }
            if (documentState.state == ClusterDbState.Remove)
            {
              //Console.WriteLine("Removing doc " + documentState.document.DocumentId);
              var db_clu = new DocumentUpdateRow() { DocumentId = documentState.document.DocumentId };
              db.Entry(db_clu).State = EntityState.Deleted;
            }
          }


          m_cluster_index.clusterStates = new Dictionary<string, OuterClusterState>();
          m_cluster_index.documentStates = new Dictionary<string, DocumentState>();
          try
          {
            db.SaveChanges();
          }
          catch (DbUpdateException e)
          {
            Console.WriteLine(e.Message);
          }
        }
      }
    }

    public void LoadClusterPool()
    {

      using (var db = new NearestClusterDbContext(_connectionOptionsBuilder.Options))
      {
        List<ClusterRow> cluster_rows = db.ClusterRows.ToList();
        foreach (var row in cluster_rows)
        {
          var new_cluster = new OuterCluster(row);
          m_cluster_index.clusterIndex.Add(Ulid.Parse(row.ClusterId), new_cluster);
        }
        List<DocumentUpdateRow> document_rows = db.DocumentUpdateRows.ToList();
        foreach (var row in document_rows)
        {
          var new_document = new DocumentUpdate(row);
          m_cluster_index.clusterIndex[Ulid.Parse(row.ClusterId)].m_document_updates.Add(new_document);
        }
        m_state.active_clusters = m_cluster_index.clusterIndex.Values.ToList();
      }
    }

    public DateTime LoadLastDocumentDatetime(string module_name)
    {
      using (var db = new NearestClusterDbContext(_connectionOptionsBuilder.Options))
      {
        PoolState pool_state = db.PoolState.Where(s => s.PoolId == module_name).FirstOrDefault();   // decide how to keep the pool records: possibility of having a pool ID, but I'm not sure if that makes sense (each scenario involves a different DB, and as such, a single row should suffice)
        if (pool_state == null)
          return new DateTime(1970, 1, 1, 0, 0, 0);
        return pool_state.LastUpdateDatetime;
      }
    }

    public void SaveLastDocumentDatetime(DateTime timestamp)
    {
      using (var db = new NearestClusterDbContext(_connectionOptionsBuilder.Options))
      {
        var existing = db.PoolState.Where(s => s.PoolId == module_name).FirstOrDefault();
        if (existing != null && timestamp > existing.LastUpdateDatetime)
        {
          existing.LastUpdateDatetime = timestamp;
          db.Update(existing);
          db.SaveChanges();
        }
        else if (existing == null)
        {
          PoolState new_state = new PoolState(this.module_name, timestamp);
          db.Add(new_state);
          db.SaveChanges();
        }
      }

    }

    public NearestClusterResult GetNearestClusters(Ulid clusterId)
    {
      NearestClusterResult nearest_clusters = new NearestClusterResult();
      OuterCluster source_cluster = m_cluster_index.clusterIndex[clusterId];
      List<NearestClusterElement> clusters_by_relevance = new List<NearestClusterElement>();
      nearest_clusters.source_cluster = clusterId.ToString();

      foreach (var target_cluster in m_cluster_index.clusterIndex.Values)
      {
        if (target_cluster.m_record_number == clusterId)
          continue;

        var cur_relevance = RankClusterPair(source_cluster, target_cluster);
        var cur_pair = new NearestClusterElement();
        cur_pair.relevance_to_cluster = cur_relevance.rank;
        cur_pair.cluster_id = target_cluster.m_record_number.ToString();
        clusters_by_relevance.Add(cur_pair);
      }
      nearest_clusters.target_clusters = clusters_by_relevance.OrderByDescending(clu => clu.relevance_to_cluster).ToList();

      return nearest_clusters;
    }

    public JObject GetStatus()
    {
      Dictionary<string, Dictionary<string, string>> result = new Dictionary<string, Dictionary<string, string>>();
      DateTime result_timestamp = new DateTime(1970, 1, 1, 0, 0, 0);

      result["global"] = new Dictionary<string, string>();

      if (config_enable_save_cluster_pool)
      {
        using (var db = new NearestClusterDbContext(_connectionOptionsBuilder.Options))
        {
          var existing = db.PoolState.Where(s => s.PoolId == module_name).FirstOrDefault();
          if (existing != null)
          {
            result_timestamp = existing.LastUpdateDatetime;
          }
        }

      }
      result["global"]["latest_updateTimestamp"] = result_timestamp.ToString("yyyy-MM-ddTHHmmss.ffffff", CultureInfo.InvariantCulture);
      return JObject.FromObject(result);

    }

    public async Task<List<ClusteringUpdate>> PutDocument(DocumentPayload document)
    {
      JsonDenseRepr feature_reprs;

      Dictionary<string, string> document_contents = new Dictionary<string, string>();
      document_contents["id"] = document.id;
      document_contents["text"] = document.text["body"];
      document_contents["title"] = document.text["title"];
      document_contents["main_repr"] = "True";
      if (config_use_title_representation) document_contents["title_repr"] = "True";
      if (config_use_paragraph_representation) document_contents["paragraph_repr"] = "True";
      if (config_use_title_paragraph_representation) document_contents["title_paragraph_repr"] = "True";

      var initial_text = document.text["body"];
      var initial_title = document.text["title"];

      List<ClusteringUpdate> l_updates = new List<ClusteringUpdate>();
      ClusteringUpdate update = new ClusteringUpdate();

      if (document.text["body"] == "" && document.text["title"] == "")
      {
        l_updates.Add(update);
        return l_updates;
      }


      m_state.candidate_i += 1;
      if (!config_use_stored_vectors)
      {
        //var watch_distil = System.Diagnostics.Stopwatch.StartNew();
        feature_reprs = await GetDistilRepresentation(document_contents, initial_text, initial_title);
        //watch_distil.Stop();
        //Console.WriteLine("GetDistilRepresentation(): " + watch_distil.ElapsedMilliseconds + " ms");
      }
      else
        feature_reprs = GetStoredVectors();


      DocumentRepresentation documentRepr = new DocumentRepresentation();
      documentRepr.payload = document;

      if (feature_reprs.main_repr != null)
        documentRepr.representation = feature_reprs.main_repr;

      if (feature_reprs.paragraph_repr != null)
      {
        documentRepr.payload.paragraph_repr = feature_reprs.paragraph_repr;
        documentRepr.payload.use_paragraph_representation = true;
      }
      if (feature_reprs.title_paragraph_repr != null)
      {
        documentRepr.payload.title_paragraph_repr = feature_reprs.title_paragraph_repr;
        documentRepr.payload.use_title_paragraph_representation = true;
      }

      List<PutDocumentResult> putDocumentResults = PutDocument(documentRepr);


      foreach (var putDocumentResult in putDocumentResults)
      {
        update = new ClusteringUpdate();
        update.document_id = document.id;
        update.group_id = document.group_id;
        update.type = "mono";
        update.cluster_id = putDocumentResult.decision.ToString();
        update.relevance_to_cluster = putDocumentResult.relevance_to_cluster;
        update.available_languages = putDocumentResult.available_languages;
        update.best_clusterindex = putDocumentResult.best_clusterindex;
        update.document_updates = putDocumentResult.document_updates;
        update.remove_from_index = putDocumentResult.remove_from_index;

        l_updates.Add(update);

      }

      return l_updates;

    }

    public JsonDenseRepr GetStoredVectors()
    {
      var distil_reprs = new JsonDenseRepr();
      //var features = new List<string>(new string[] { "main_repr", "paragraph_repr", "title_paragraph_repr" });

      using (Stream stream = File.Open(config_stored_vectors_path, FileMode.Open)) //replace string w/ appsettings parameter
      {
        BinaryReader binary_reader = new BinaryReader(stream);
        stream.Position = stream_position;    //update this after the new reads alongside the final distil_reprs object

        distil_reprs.main_repr = ParseDenseVector(binary_reader);
        distil_reprs.paragraph_repr = ParseDenseVector(binary_reader);
        distil_reprs.title_paragraph_repr = ParseDenseVector(binary_reader);


        this.stream_position = stream.Position;
      }

      return distil_reprs;
    }

    public DenseVector ParseDenseVector(BinaryReader binary_reader)
    {

      DenseVector dense_vector = new DenseVector();

      for (int i = 0; i < dense_vector.dense_vector.Count; i += 1)
      {
        double el = binary_reader.ReadDouble();
        dense_vector.dense_vector[i] = el;
      }

      return dense_vector;
    }


    public List<PutDocumentResult> PutDocument(DocumentRepresentation documentRepr)
    {
      UpdateOldestCluster();

      List<PutDocumentResult> results = new List<PutDocumentResult>();
      PutDocumentResult putDocumentResult = new PutDocumentResult();
      ClusterResult clusterResult = CreateFreeCluster(documentRepr.payload, documentRepr.representation);
      putDocumentResult.decision = clusterResult.cluster_id;
      putDocumentResult.relevance_to_cluster = clusterResult.relevance_to_cluster;
      putDocumentResult.available_languages = clusterResult.available_languages;

      DocumentUpdate doc_update = new DocumentUpdate();
      putDocumentResult.document_updates = new List<DocumentUpdate>();
      doc_update.document_id = documentRepr.payload.id;
      doc_update.language = documentRepr.payload.language;
      doc_update.group_id = documentRepr.payload.group_id;
      doc_update.forced_label = documentRepr.payload.forced_label;
      doc_update.timestamp = documentRepr.payload.timestamp;
      doc_update.update_timestamp = documentRepr.payload.update_timestamp;
      doc_update.similarity = 2;

      putDocumentResult.document_updates.Add(doc_update);
      results.Add(putDocumentResult);

      if (!clusterResult.is_decided)
      {
        BestClusterInfo bestCluster = FindHighestRankingCluster(documentRepr.payload, documentRepr.representation);

        if (!bestCluster.found)
        {
          clusterResult = CreateFreeCluster(documentRepr.payload, documentRepr.representation, true);
        }

        var mergeDecisionResult = DoMergeDecision(documentRepr.payload, bestCluster.clusterindex, bestCluster.rank, bestCluster.similarity,
                documentRepr.representation, clusterResult.relevance_to_cluster, putDocumentResult.decision);
        results = mergeDecisionResult;

        if (config_check_cluster_merge && !config_use_forced_label)
        {
          results.AddRange(CheckClusterMerge(m_state.active_clusters.ElementAt(bestCluster.clusterindex)));

        }

      }

      if (config_enable_save_cluster_pool && (m_state.candidate_i % save_cluster_pool_interval == 0))
      {
        var watch = System.Diagnostics.Stopwatch.StartNew();
        //var t = Task.Run(() => SaveClusterPool());
        SaveClusterPool();

        watch.Stop();
        Console.WriteLine("SaveClusterPool(): " + watch.ElapsedMilliseconds + " ms");

        SaveLastDocumentDatetime(documentRepr.payload.update_timestamp);
      }

      return results;
    }


    public async Task<JsonDenseRepr> GetDistilRepresentation(Dictionary<string, string> document_contents, string initial_text, string initial_title)
    {

      HttpContent post_content = new StringContent(JsonConvert.SerializeObject(document_contents), Encoding.UTF8);
      string url = config_distil_service_url;
      string result_string;
      const int sleepms = 600000; // 10 min per sleep
      var retry = true;

      while (retry)
      {
        try
        {
          //var distil_reprs = new Dictionary<string, DenseVector>();
          HttpResponseMessage distil_result = await StaticHttpClient.Instance.PostAsync(url, post_content);
          result_string = await distil_result.Content.ReadAsStringAsync();
          //var result_json = JsonRazor.JsonToken.Read(result_string);
          //JObject result_json = JObject.Parse(result_string);
          //var feature_reprs = Deserializer.Consume<JsonRepr>(result_string);
          var feature_reprs = JsonConvert.DeserializeObject<JsonRepr>(result_string);
          JsonDenseRepr distil_reprs = new JsonDenseRepr(feature_reprs);

          //var feature_reprs = JsonConvert.DeserializeObject<Dictionary<string, double[]>>(result_json.ToString());

          //change configuration bool to be defined in the beginning
          if (config_generate_new_stored_vectors)
          {
            using (Stream stream = File.Open(config_stored_vectors_path, FileMode.Append))
            {
              BinaryWriter binary_writer = new BinaryWriter(stream);

              foreach (var f in feature_reprs.GetReprs())
              {
                foreach (var el in f)
                {
                  binary_writer.Write(el);
                }
              }
              binary_writer.Flush();
            }
          }

          return distil_reprs;

        }
        catch (TaskCanceledException e)
        {
          Thread.Sleep(sleepms);
          logger.LogInformation("Exception posting to url " + url);
          logger.LogInformation($"Task canceled (Client canceled request or Http Timeout): {e.Message}. Sleep and retry forever...");
          retry = true;
          continue;
        }
        catch (HttpRequestException e)
        {
          Thread.Sleep(sleepms);
          logger.LogInformation("Exception posting to url " + url);
          logger.LogInformation($"Task canceled (Client canceled request or Http Timeout): {e.Message}. Sleep and retry forever...");
          retry = true;
          continue;
        }
        catch (Exception e)
        {
          /*
          if (String.IsNullOrWhiteSpace(document_contents["title"]) && String.IsNullOrWhiteSpace(document_contents["text"]))
          {
            logger.LogInformation("Exception posting to url " + url + "; e: " + e.Message);
            logger.LogInformation("Doc. Title: " + document_contents["title"]);
            logger.LogInformation("Doc. Text: " + document_contents["text"]);
            logger.LogInformation("Doc. Id: " + document_contents["id"]);
            logger.LogInformation("Original Title: " + initial_title);
            logger.LogInformation("Original Text: " + initial_text);
            return null;
          }
          */
          logger.LogInformation("Exception posting to url " + url + "; e: " + e.Message);
          logger.LogInformation("Document ID: " + document_contents["id"]);
          return null;

        }
      }
      return null;
    }

    public BestClusterInfo FindHighestRankingCluster(DocumentPayload document,
                        DenseVector document_hfv)
    {
      BestClusterInfo cluster_info = new BestClusterInfo();
      double forced_best_rank = 0.0;
      int forced_best_clusterindex = -1;
      int best_clusterindex = -1;
      double best_rank = double.MinValue;
      bool always_merge = document.allways_merges_to_cluster;
      qid += 1;

      List<ClusterRankResult> rank_examples = new List<ClusterRankResult>();
      List<ClusterRankResult> merge_examples = new List<ClusterRankResult>();

      object lock_cluster = new object();
      object lock_qid = new object();

      Parallel.ForEach(m_state.active_clusters, cluster =>
      {

        ClusterRankResult cr_result = RankCluster(document, document_hfv, cluster, m_classifier);

        lock (lock_cluster)
        {
          if (cr_result.rank > best_rank)
          {
            best_rank = cr_result.rank;
            best_clusterindex = m_state.active_clusters.IndexOf(cluster);
            cluster_info.similarity = cr_result.similarity;
          }
          if (document.use_forced_label && document.forced_label == cluster.forced_label)
          {
            forced_best_rank = double.MaxValue;
            forced_best_clusterindex = m_state.active_clusters.IndexOf(cluster);
          }
          if (config_generate_rank_examples)
          {
            rank_examples.Add(cr_result);
          }
        }
      });


      /*
      foreach (OuterCluster cluster in m_state.active_clusters)
      {
        ++cluster_i;
        //bool decided = false;

        if (document.use_forced_label && document.forced_label == cluster.forced_label)
        {
          forced_best_rank = double.MaxValue;
          forced_best_clusterindex = cluster_i;
          //decided = true;
        }

        if (document.use_forced_label && !always_merge)
        {
          if (config_generate_rank_examples)
          {
            ClusterRankResult gen_rank = RankCluster(document, document_hfv, cluster, m_classifier);
            rank_examples.Add(gen_rank);
            min_rank = gen_rank.similarity.embedding_all < min_rank ? gen_rank.similarity.embedding_all : min_rank;
            max_rank = gen_rank.similarity.embedding_all > max_rank ? gen_rank.similarity.embedding_all : max_rank;
          }
          else if (config_generate_merge_examples)
          {
            ClusterRankResult gen_rank = RankCluster(document, document_hfv, cluster, m_classifier);
            merge_examples.Add(gen_rank);
          }
          //else if (decided)
          //  break;
          //continue;
        }

        ClusterRankResult cr_result = RankCluster(document, document_hfv, cluster, m_classifier);

        if (cr_result.rank > best_rank)
        {
          best_rank = cr_result.rank;
          best_clusterindex = cluster_i;
          cluster_info.similarity = cr_result.similarity;
        }
      //});
      }

      */
      if (config_generate_merge_examples && document.use_forced_label)
      {
        if (config_generate_extra_neg_merge_examples)
        {
          GenerateMergeExamples(merge_examples, document.forced_label, document.group_id);
          GenerateMergeExamples(merge_examples, document.forced_label, "clusters_multi");
        }
        else
        {
          ClusterRankResult cr_merge = new ClusterRankResult();
          cr_merge.cluster_forced_label = cluster_info.cluster_forced_label;
          cr_merge.rank = cluster_info.rank;
          cr_merge.similarity = cluster_info.similarity;

          WriteMergeExample(cr_merge, document.forced_label, document.group_id);
          WriteMergeExample(cr_merge, document.forced_label, "clusters_multi");
        }
      }

      if (document.use_forced_label)
      {
        best_rank = forced_best_rank;
        best_clusterindex = forced_best_clusterindex;
      }

      if (config_generate_rank_examples && document.use_forced_label && best_clusterindex != -1 && m_cluster_index.clusterIndex.Count > 1)
      {
        var correct_result = rank_examples.Where(cr => cr.cluster_forced_label == document.forced_label).First();
        GenerateRankExamples(correct_result, document.forced_label, document.group_id);
        GenerateRankExamples(correct_result, document.forced_label, "clusters_multi");
        // foreach (var cr_result in rank_examples)
        foreach (var cr_result in rank_examples.Where(cr => cr.cluster_forced_label != document.forced_label).OrderByDescending(cr => cr.rank).Take(50))
        {
          GenerateRankExamples(cr_result, document.forced_label, document.group_id);
          GenerateRankExamples(cr_result, document.forced_label, "clusters_multi");
        }
      }


      if (!document.use_forced_label && (best_clusterindex == -1 ||
                         best_clusterindex >=
                         m_state.active_clusters.Count))
      {
        logger.LogInformation("Failed finding cluster for document : " + document.UniqueId());
        cluster_info.found = false;
      }
      else
      {
        cluster_info.found = true;
      }

      if (config_evaluate_cluster_recall)
      {
        var ordered_rank_examples = rank_examples.OrderByDescending(x => x.rank).ToList();
        cluster_info.clusterindex = ordered_rank_examples.FindIndex(x => x.cluster_forced_label == document.forced_label);

      }
      else
      {
        cluster_info.clusterindex = best_clusterindex;
      }

      cluster_info.rank = best_rank;
      return cluster_info;
    }


    public void GenerateRankExamples(ClusterRankResult cr_result, string document_label, string group_id)
    {
      using (StreamWriter rankingWriter = File.AppendText(configuration[$"TrainEval:RankingDataOutputPath"] + "\\" + group_id + "_" + boot_time + ".svmin"))
      {
        var i = 0;
        var label = document_label == cr_result.cluster_forced_label ? 1 : 0;
        double[] feature_mask = new double[] { 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0 };  //no title_repr and target_density
        var feature_array = cr_result.similarity.GetFeatureArray();
        //string[] features_ordered = { "embedding_all", "NEWEST_TS", "OLDEST_TS", "RELEVANCE_TS", "paragraph_repr", "title_paragraph_repr", "paragraph_centroid", "title_paragraph_centroid", "cluster_density", "mean_similarity"};
        rankingWriter.Write(label + " qid:" + qid + " ");

        for (var f = 0; f < feature_mask.Length; f++)
        {
          if (feature_mask[f] == 1)
          {
            i += 1;
            rankingWriter.Write(i.ToString() + ":" + feature_array[f].ToString("0.######", inv_c) + " ");
          }
        }
        rankingWriter.Write("\n");
      }
    }

    public void WriteMergeExample(ClusterRankResult cluster_info, string document_label, string group_id, string extra_neg = "")
    {
      using (StreamWriter rankingWriter = File.AppendText(configuration[$"TrainEval:MergeDataOutputPath"] + "\\" + group_id + "_" + boot_time + extra_neg + ".svmin"))
      {
        var i = 0;
        var label = document_label == cluster_info.cluster_forced_label ? 1 : 0;

        // order is { embedding_all, newest_ts, oldest_ts, relevance_ts, title_repr, paragraph_repr, title_paragraph_repr, paragraph_centroid, title_paragraph_centroid, cluster_density, mean_similarity, target_density}

        // we want { "embedding_all", "NEWEST_TS", "OLDEST_TS", "RELEVANCE_TS", "paragraph_repr", "title_paragraph_repr", "paragraph_centroid", "title_paragraph_centroid", "cluster_density", "mean_similarity" }

        double[] feature_mask = new double[] { 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0 };  //do not get title_repr or target_density

        var feature_array = cluster_info.similarity.GetFeatureArray();

        //string[] features_ordered = { "embedding_all", "NEWEST_TS", "OLDEST_TS", "RELEVANCE_TS", "paragraph_repr", "title_paragraph_repr", "paragraph_centroid", "title_paragraph_centroid", "cluster_density", "mean_similarity"};
        rankingWriter.Write(label + " ");
        //foreach(string key in cluster_info.similarity.Keys)

        for (var f = 0; f < feature_mask.Length; f++)
        {
          if (feature_mask[f] == 1)
          {
            i += 1;
            rankingWriter.Write(i.ToString() + ":" + feature_array[f].ToString("0.######", inv_c) + " ");
          }
        }
        rankingWriter.Write("\n");
      }
    }

    public void WriteClusterJoinExample(FeatureCollection sample, int label)
    {
      using (StreamWriter rankingWriter = File.AppendText(configuration[$"TrainEval:MergeDataOutputPath"] + "\\clusters_multi_" + boot_time + "_clusterjoin.svmin"))
      {
        var i = 0;
        //string[] features_ordered = { "embedding_all", "NEWEST_TS", "OLDEST_TS", "RELEVANCE_TS", "paragraph_repr", "title_paragraph_repr", "paragraph_centroid", "title_paragraph_centroid", "cluster_density", "target_density"};
        double[] feature_mask = new double[] { 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1 };  //do not get title_repr or mean_similarity

        var feature_array = sample.GetFeatureArray();

        rankingWriter.Write(label + " ");

        for (var f = 0; f < feature_mask.Length; f++)
        {
          if (feature_mask[f] == 1)
          {
            i += 1;
            rankingWriter.Write(i.ToString() + ":" + feature_array[f].ToString("0.######", inv_c) + " ");
          }
        }
        rankingWriter.Write("\n");

        /*
                foreach (string key in features_ordered)
                {
                  if (sample.Keys.Contains(key))
                  {
                    i += 1;
                    rankingWriter.Write(i.ToString() + ":" + sample[key].ToString("0.######", inv_c) + " ");
                  }
                }
                rankingWriter.Write("\n");
                */
      }

    }


    public void GenerateMergeExamples(List<ClusterRankResult> merge_examples, string document_label, string group_id)
    {
      List<ClusterRankResult> sorted_examples = merge_examples.OrderByDescending(e => e.rank).ToList();
      WriteMergeExample(sorted_examples[0], document_label, group_id);
      WriteMergeExample(sorted_examples[0], document_label, group_id, ".neg");

      if (config_random_negative_merge_samples)
      {
        if (sorted_examples[0].cluster_forced_label != document_label)
        {
          for (var i = 1; i < sorted_examples.Count; i++)
          {
            if (sorted_examples[i].cluster_forced_label == document_label)  //find the placement of the gold label
            {
              WriteMergeExample(sorted_examples[i], document_label, group_id, ".neg");
              break;
            }
          }
        }
        else if (sorted_examples.Count > 1)
        {
          var selected_sample = new Random().Next(1, Math.Min(10, sorted_examples.Count - 1));
          WriteMergeExample(sorted_examples[selected_sample], document_label, group_id, ".neg");
        }
      }
      else
      {
        for (var i = 1; i < sorted_examples.Count; i++)
        {
          if ((sorted_examples.Count > 1 && sorted_examples[0].cluster_forced_label == document_label) ||
            (sorted_examples[i].cluster_forced_label == document_label))  //get either the first negative label (if the best example was gold), or find the placement of the gold label
          {
            WriteMergeExample(sorted_examples[i], document_label, group_id, ".neg");
            break;
          }
        }
      }
    }


    public ClusterRankResult RankCluster(DocumentPayload document,
                  DenseVector document_hfv,
                  Cluster cluster,
                  ClassifierModel rank_classifier)
    {
      ClusterRankResult result = new ClusterRankResult();
      DenseVector main_repr = cluster.GetMainRepresentation();

      var cos_sim = main_repr.CosineSimilarity(document_hfv) + 1;   //force similarity to [0,2] range since cos sim has it as [-1, 1] initially
      FeatureCollection similarity_features = new FeatureCollection();
      similarity_features.embedding_all = cos_sim;

      //var similarity_features = main_repr.FeatureCosineSimilarity(document_hfv);
      if (config_use_timestamp_features)
      {
        DateTime cluster_relevance_timestamp = new DateTime(TimeSpan.FromHours(cluster.RelevanceStamp(relevance_stamp_zscore)).Ticks
                            + new DateTime(1970, 1, 1, 0, 0, 0).Ticks);  //hotfix; think about how to make this clean

        AddTimestampFeatures(document.timestamp,
                   gaussian_std_dev,
                   cluster_relevance_timestamp,
                   cluster.m_newest_timestamp,
                   cluster.m_oldest_timestamp,
                   similarity_features);

      }

      if (config_use_title_representation && document.text["title"] != "")
        similarity_features.title_repr = cluster.title_centroid.CosineSimilarity(document.title_repr) + 1;   //force similarity to [0,2] range since cos sim has it as [-1, 1] initially 

      if (config_use_paragraph_representation && document.paragraph_repr != null)
      {
        similarity_features.paragraph_repr = cluster.paragraph_centroid.CosineSimilarity(document.paragraph_repr) + 1;   //force similarity to [0,2] range since cos sim has it as [-1, 1] initially 
        similarity_features.paragraph_centroid = main_repr.CosineSimilarity(document.paragraph_repr) + 1;
      }

      if (config_use_title_paragraph_representation && document.title_paragraph_repr != null)
      {
        similarity_features.title_paragraph_repr = cluster.title_paragraph_centroid.CosineSimilarity(document.title_paragraph_repr) + 1;   //force similarity to [0,2] range since cos sim has it as [-1, 1] initially 
        similarity_features.title_paragraph_centroid = main_repr.CosineSimilarity(document.title_paragraph_repr) + 1;
      }

      if (config_use_cluster_density)
      {
        similarity_features.cluster_density = cluster.GetBinFeature();
      }

      if (config_use_mean_similarity)
        similarity_features.mean_similarity = cluster.GetMeanSimilarity() + 1;

      result.cluster_forced_label = cluster.forced_label;
      result.rank = rank_classifier.Score(similarity_features);
      result.similarity = similarity_features;
      return result;
    }


    public ClusterRankResult RankClusterPair(Cluster source_cluster,
                                  Cluster target_cluster)
    {
      ClusterRankResult result = new ClusterRankResult();
      DenseVector source_repr = source_cluster.GetMainRepresentation();
      DenseVector target_repr = target_cluster.GetMainRepresentation();

      var cos_sim = source_repr.CosineSimilarity(target_repr) + 1;  //force similarity to [0,2] range since cos sim has it as [-1, 1] initially
      FeatureCollection similarity_features = new FeatureCollection();
      similarity_features.embedding_all = cos_sim;


      if (config_use_timestamp_features)
      {
        AddTimestampFeaturesPairwiseCluster(source_cluster,
                   target_cluster,
                   gaussian_std_dev,
                   similarity_features);
      }


      if (config_use_title_representation && source_cluster.title_centroid != null && target_cluster.title_centroid != null)
        similarity_features.title_repr = source_cluster.title_centroid.CosineSimilarity(target_cluster.title_centroid) + 1;   //force similarity to [0,2] range since cos sim has it as [-1, 1] initially 

      if (config_use_paragraph_representation && source_cluster.paragraph_centroid != null && target_cluster.paragraph_centroid != null)
      {
        similarity_features.paragraph_repr = source_cluster.paragraph_centroid.CosineSimilarity(target_cluster.paragraph_centroid) + 1;   //force similarity to [0,2] range since cos sim has it as [-1, 1] initially 
        similarity_features.paragraph_centroid = source_repr.CosineSimilarity(target_cluster.paragraph_centroid) + 1;
      }

      if (config_use_title_paragraph_representation && source_cluster.title_paragraph_centroid != null && target_cluster.title_paragraph_centroid != null)
      {
        similarity_features.title_paragraph_repr = source_cluster.title_paragraph_centroid.CosineSimilarity(target_cluster.title_paragraph_centroid) + 1;   //force similarity to [0,2] range since cos sim has it as [-1, 1] initially 
        similarity_features.title_paragraph_centroid = source_repr.CosineSimilarity(target_cluster.title_paragraph_centroid) + 1;
      }

      if (config_use_cluster_density)
        similarity_features.cluster_density = source_cluster.GetBinFeature();

      if (config_use_mean_similarity)
        similarity_features.mean_similarity = source_cluster.GetMeanSimilarity() + 1;

      result.rank = m_classifier.Score(similarity_features);
      result.similarity = similarity_features;
      return result;
    }

    public ClusterResult CreateFreeCluster(DocumentPayload document,
                    DenseVector document_hfv,
                    bool force = false)
    {
      ClusterResult result = new ClusterResult();
      double relevance_to_cluster = -1.0;
      var available_languages = new HashSet<string>();
      if (force || m_state.candidate_i < config_k_first_clusters)
      {
        OuterCluster new_cluster = CreateCluster();

        relevance_to_cluster = new_cluster.AddPoint(document_hfv,
                              document,
                              0.0,
                              relevance_to_cluster,
                              true).relevance;
        available_languages = new_cluster.available_languages;

        m_cluster_index.InsertClusterState(new OuterClusterState(ClusterDbState.Add, new_cluster));

        var doc_update_row = new DocumentUpdateRow(new_cluster.m_record_number.ToString(), new_cluster.m_document_updates.First());
        m_cluster_index.InsertDocumentState(new DocumentState(ClusterDbState.Add, doc_update_row));
        //m_cluster_index.clustersToAdd.Add(new_cluster);

        /*
                if (config_use_title_representation && document.text["title"] != "")
                  new_cluster.title_centroid.Add(document.title_repr);

                if (config_use_paragraph_representation)
                  new_cluster.paragraph_centroid.Add(document.paragraph_repr);

                if (config_use_title_paragraph_representation)
                  new_cluster.title_paragraph_centroid.Add(document.title_paragraph_repr);
        */

        m_state.active_clusters.Add(new_cluster);

        PostProcessDecision(document, new_cluster);

        var new_index = m_state.active_clusters.Count - 1;
        var stamp = new_cluster.RelevanceStamp(relevance_stamp_zscore);
        if (m_oldest_cluster_index == -1 || stamp < m_oldest_cluster_relevance_stamp)
        {
          m_oldest_cluster_relevance_stamp = stamp;
          m_oldest_cluster_index = new_index;
        }
        result.is_decided = true;
        result.cluster_id = new_cluster.m_record_number;
        result.available_languages = available_languages;
      }
      result.relevance_to_cluster = relevance_to_cluster;
      return result;
    }

    public void PostProcessDecision(DocumentPayload document,
                    Cluster cluster)
    {
      UpdateOldestCluster();

      if (document.use_forced_label && String.IsNullOrEmpty(cluster.forced_label))
        cluster.forced_label = document.forced_label;
    }

    public void UpdateOldestCluster()
    {
      int cluster_k = -1;
      foreach (var cluster in m_state.active_clusters)
      {
        ++cluster_k;
        var stamp = cluster.RelevanceStamp(relevance_stamp_zscore);
        if (m_oldest_cluster_index == -1 || stamp < m_oldest_cluster_relevance_stamp)
        {
          m_oldest_cluster_relevance_stamp = stamp;
          m_oldest_cluster_index = cluster_k;
        }
      }
    }

    public List<PutDocumentResult> DoMergeDecision(DocumentPayload document,
                  int best_clusterindex,
                  double best_rank,
                  FeatureCollection best_similarity,
                  DenseVector document_hfv,
                  double relevance_to_cluster,
                  Ulid decision)
    {
      bool force_merge = false;
      bool force_new = false;
      AddPointRes add_point_res = new AddPointRes();
      add_point_res.relevance = relevance_to_cluster;
      var available_languages = new HashSet<string>();
      List<PutDocumentResult> results = new List<PutDocumentResult>();

      if (document.use_forced_label)
      {
        if (best_clusterindex == -1)
          force_new = true;
        else
          force_merge = true;
      }
      else if (document.allways_merges_to_cluster)
      {
        force_merge = true;
      }

      OuterCluster best_cluster = null;

      if (best_clusterindex != -1)
      {
        best_cluster = m_state.active_clusters.ElementAt(best_clusterindex);
      }

      if (!force_new && (force_merge || AcceptDocumentCriterion(best_cluster, best_similarity, best_rank, m_merge_classifier)) && best_cluster != null)
      {
        add_point_res = best_cluster.AddPoint(document_hfv,
                    document,
                    best_rank,
                    relevance_to_cluster,
                    !document.allways_merges_to_cluster && !document.do_not_modify_centroid);

        //commenting to test w/merged microclusters
        //if (best_cluster.m_micro_clusters.Count == microclusters_limit && !document.use_forced_label)
        //  results.AddRange(CheckDominoToppling(best_cluster.m_micro_clusters, best_clusterindex, best_cluster.m_centroid));
        //else
        //  toppling_occurrences.Add(0);

        //results.AddRange(CheckDominoToppling(best_cluster.m_micro_clusters, best_clusterindex, best_cluster.m_centroid));

        available_languages = best_cluster.available_languages;

        PostProcessDecision(document, best_cluster);
        decision = best_cluster.m_record_number;
        m_cluster_index.InsertClusterState(new OuterClusterState(ClusterDbState.Update, best_cluster));
        //m_cluster_index.clustersToUpdate.Add(best_cluster);
        //m_cluster_index.documentsToAdd.Add(new DocumentUpdateRow(best_cluster.m_record_number.ToString(), best_cluster.m_document_updates.Last()));
      }
      else
      {
        if (m_state.active_clusters.Count >= int.Parse(configuration[$"Main:ClusterOptions:ClusterActivePoolsize"], inv_c))
        {
          var archive_index = m_state.oldest_cluster.index;
          if (archive_index >= 0 && archive_index < m_state.active_clusters.Count)
          {
            var archive_cluster = m_state.active_clusters[archive_index];
            var new_inactive_index = m_state.inactive_clusters.Count;
            m_state.inactive_clusters.Add(archive_cluster);
            m_state.oldest_cluster.index = -1;
            m_state.active_clusters.RemoveAt(archive_index);
            //m_cluster_index.clustersToRemove.Add(archive_cluster);

            m_cluster_index.clusterIndex.Remove(archive_cluster.m_record_number);
            m_cluster_index.InsertClusterState(new OuterClusterState(ClusterDbState.Remove, archive_cluster));
            foreach (var arch_doc in archive_cluster.m_document_updates)
            {
              var doc_to_remove = new DocumentUpdateRow(archive_cluster.m_record_number.ToString(), arch_doc);
              m_cluster_index.InsertDocumentState(new DocumentState(ClusterDbState.Remove, doc_to_remove));

              //m_cluster_index.documentIdsToRemove.Add(arch_doc.document_id);
            }
          }
        }

        best_cluster = CreateCluster();
        add_point_res = best_cluster.AddPoint(document_hfv,
                   document,
                   0.0,
                   relevance_to_cluster,
                   true);
        available_languages = best_cluster.available_languages;

        m_state.active_clusters.Add(best_cluster);
        int new_cluster_index = m_state.active_clusters.Count - 1;
        PostProcessDecision(document, best_cluster);

        var stamp = best_cluster.RelevanceStamp(relevance_stamp_zscore);
        if (m_state.oldest_cluster.index == -1 || stamp < m_state.oldest_cluster.relevance_stamp)
        {
          m_state.oldest_cluster.relevance_stamp = stamp;
          m_state.oldest_cluster.index = m_state.active_clusters.Count - 1;

        }

        decision = best_cluster.m_record_number;

        m_cluster_index.InsertClusterState(new OuterClusterState(ClusterDbState.Add, best_cluster));
        //m_cluster_index.clustersToAdd.Add(best_cluster);

        //m_cluster_index.documentsToAdd.Add(new DocumentUpdateRow(best_cluster.m_record_number.ToString(), best_cluster.m_document_updates.Last()));
      }


      var doc_update_row = new DocumentUpdateRow(best_cluster.m_record_number.ToString(), best_cluster.m_document_updates.Last());
      m_cluster_index.InsertDocumentState(new DocumentState(ClusterDbState.Add, doc_update_row));

      PutDocumentResult result = new PutDocumentResult();
      result.decision = decision;
      result.relevance_to_cluster = add_point_res.relevance;
      result.available_languages = available_languages;
      result.best_clusterindex = best_clusterindex;

      DocumentUpdate doc_update = new DocumentUpdate();
      doc_update.document_id = document.id;
      doc_update.language = document.language;
      doc_update.group_id = document.group_id;
      doc_update.forced_label = document.forced_label;
      doc_update.similarity = (best_similarity != null ? best_similarity.embedding_all : 1);  //if this is the first document, it'll be a single-doc cluster
      doc_update.timestamp = document.timestamp;
      doc_update.update_timestamp = document.update_timestamp;

      //doing this to maroscate the localf1 function; populate the possible labels w/ list of clusters in order to retrieve them quickly
      if (!label_index.ContainsKey(document.forced_label))
        label_index[document.forced_label] = new List<OuterCluster>();
      if (!label_index[document.forced_label].Contains(best_cluster))
        label_index[document.forced_label].Add(best_cluster);

      result.document_updates = new List<DocumentUpdate>();
      result.document_updates.Add(doc_update);
      results.Add(result);

      var documentRepr = new DocumentRepresentation();
      documentRepr.representation = document_hfv;
      documentRepr.payload = document;

      return results;
    }

    public OuterCluster CreateCluster()
    {
      OuterCluster new_cluster = new OuterCluster();
      Ulid new_ulid = Ulid.NewUlid(DateTimeOffset.Now);
      new_cluster.m_record_number = new_ulid;
      m_cluster_index.clusterIndex[new_ulid] = new_cluster;

      return new_cluster;

    }

    public bool AcceptDocumentCriterion(Cluster cluster, FeatureCollection best_similarity, double best_rank, ClassifierModel merge_classifier)
    {
      if (merge_classifier.is_loaded)
        return merge_classifier.Score(best_similarity) > merge_thr;

      return best_rank > merge_thr;
    }

    public void AddTimestampFeatures(DateTime timestamp,
                     double gaussian_stddev,
                     DateTime cluster_relevance_timestamp,
                     DateTime cluster_newest_timestamp,
                     DateTime cluster_oldest_timestamp,
                     FeatureCollection similarity_features)
    {
      similarity_features.newest_ts = NormalizedGaussian(0, gaussian_stddev, (timestamp - cluster_newest_timestamp).TotalDays);
      similarity_features.oldest_ts = NormalizedGaussian(0, gaussian_stddev, (timestamp - cluster_oldest_timestamp).TotalDays);
      similarity_features.relevance_ts = NormalizedGaussian(0, gaussian_stddev, (timestamp - cluster_relevance_timestamp).TotalDays);
    }

    public void AddTimestampFeaturesPairwiseCluster(Cluster source_cluster,
                     Cluster target_cluster,
                     double gaussian_stddev,
                     FeatureCollection similarity_features)
    {
      DateTime source_cluster_relevance_timestamp = new DateTime(TimeSpan.FromHours(source_cluster.RelevanceStamp(relevance_stamp_zscore)).Ticks
                          + new DateTime(1970, 1, 1, 0, 0, 0).Ticks);
      DateTime target_cluster_relevance_timestamp = new DateTime(TimeSpan.FromHours(target_cluster.RelevanceStamp(relevance_stamp_zscore)).Ticks
                          + new DateTime(1970, 1, 1, 0, 0, 0).Ticks);

      similarity_features.newest_ts = NormalizedGaussian(0, gaussian_stddev, (source_cluster.m_newest_timestamp - target_cluster.m_newest_timestamp).TotalDays);
      similarity_features.oldest_ts = NormalizedGaussian(0, gaussian_stddev, (source_cluster.m_oldest_timestamp - target_cluster.m_oldest_timestamp).TotalDays);
      similarity_features.relevance_ts = NormalizedGaussian(0, gaussian_stddev, (source_cluster_relevance_timestamp - target_cluster_relevance_timestamp).TotalDays);
    }
    public double NormalizedGaussian(double mean,
                     double stddev,
                     double x)
    {
      return Math.Exp(-((x - mean) * (x - mean)) / (2 * stddev * stddev));
    }

    public List<PutDocumentResult> CheckClusterMerge(OuterCluster cluster_i)
    {
      List<Ulid> merged_clusters = new List<Ulid>();
      List<OuterCluster> final_clusters = new List<OuterCluster>();
      List<PutDocumentResult> updates = new List<PutDocumentResult>();

      List<ClusterMergeResult> merge_candidates = new List<ClusterMergeResult>();
      bool has_merged = false;

      Parallel.For(0, m_state.active_clusters.Count, j =>
      {
        var cluster_j = m_state.active_clusters[j];

        if (cluster_i.m_record_number != cluster_j.m_record_number && !merged_clusters.Contains(cluster_j.m_record_number))
        {
          var rank_result = RankClusterPair(cluster_i, cluster_j);
          var sim_ij = rank_result.similarity;
          var merge_rank = m_merge_classifier.Score(sim_ij);
          rank_result.similarity.cluster_density = cluster_i.GetBinFeature();
          rank_result.similarity.target_density = cluster_j.GetBinFeature();


          ClusterMergeResult result = new ClusterMergeResult();
          result.rank_result = rank_result;
          result.score = merge_rank;
          result.cluster = cluster_j;

          lock (merge_candidates)
          {
            merge_candidates.Add(result);
          }
        }
        //continue;

        //if (merged_clusters.Contains(cluster_j.m_record_number))
        //continue;


      });
      /*
      for (var j = 0; j < m_state.active_clusters.Count; j++)
      {
        var cluster_j = m_state.active_clusters[j];

        if (cluster_i.m_record_number == cluster_j.m_record_number)
          continue;

        if (merged_clusters.Contains(cluster_j.m_record_number))
          continue;

        var rank_result = RankClusterPair(cluster_i, cluster_j);
        var sim_ij = rank_result.similarity;
        var merge_rank = m_merge_classifier.Score(sim_ij);
        rank_result.similarity.cluster_density = cluster_i.GetBinFeature();
        rank_result.similarity.target_density = cluster_j.GetBinFeature();


        ClusterMergeResult result = new ClusterMergeResult();
        result.rank_result = rank_result;
        result.score = merge_rank;
        result.cluster = cluster_j;
        merge_candidates.Add(result);
      }
      */

      if (config_generate_clusterjoin_examples)
      {

        bool had_join = false;
        List<ClusterJoinSample> samples = new List<ClusterJoinSample>();
        foreach (var result in merge_candidates.OrderByDescending(cr => cr.score))
        {
          var clu_comparison = RankClusterPair(cluster_i, result.cluster).similarity;
          clu_comparison.cluster_density = cluster_i.GetBinFeature();
          clu_comparison.target_density = result.cluster.GetBinFeature();
          //clu_comparison["merge_score"] = result.score + 3;
          //clu_comparison["mean_similarity_i"] = cluster_i.GetMeanSimilarity() + 1;
          //clu_comparison["mean_similarity_j"] = result.cluster.GetMeanSimilarity() + 1;
          //var separate_f1 = GetLocalF1(new List<OuterCluster>(new OuterCluster[] { cluster_i, result.cluster }));
          //var joined_f1 = GetLocalF1(new List<OuterCluster>(new OuterCluster[] { cluster_i, result.cluster }), true);
          var separate_f1 = GetAltLocalF1(new List<Cluster>(new Cluster[] { cluster_i, result.cluster }));
          var joined_f1 = GetAltJoinLocalF1(new List<Cluster>(new Cluster[] { cluster_i, result.cluster }));
          //if (m_merge_classifier.Score(clu_comparison.similarity) > 0 && joined_f1 > separate_f1) //do this due to the merge updates on cluster_i
          if (joined_f1 > separate_f1)
          {
            cluster_i.JoinCluster(result.cluster);
            merged_clusters.Add(result.cluster.m_record_number);
            m_state.active_clusters.Remove(result.cluster);
            had_join = true;


            //the following section propagates the updates to the evaluator

            PutDocumentResult update = new PutDocumentResult();
            update.decision = result.cluster.m_record_number;
            update.relevance_to_cluster = joined_f1;
            update.available_languages = result.cluster.available_languages;
            update.best_clusterindex = -1;

            update.document_updates = result.cluster.m_document_updates;
            update.remove_from_index = true;
            updates.Add(update);


            update = new PutDocumentResult();
            update.decision = cluster_i.m_record_number;
            update.relevance_to_cluster = joined_f1;
            update.available_languages = cluster_i.available_languages;
            update.best_clusterindex = -1;

            update.document_updates = result.cluster.m_document_updates;
            updates.Add(update);


          }
          samples.Add(new ClusterJoinSample(clu_comparison, joined_f1 > separate_f1 ? 1 : 0));
        }
        if (had_join)
        {
          int num_writes = 0;
          bool found_join = false;
          foreach (var sample in samples)
          {
            WriteClusterJoinExample(sample.similarity, sample.label);
            if (sample.label == 1)
              found_join = true;
            if (found_join)
              num_writes += 1;
            if (num_writes == 10)
              break;
          }
        }
      }
      else
      {

        foreach (var result in merge_candidates.OrderByDescending(cr => cr.score))
        {
          // Note: RankClusterPair() has to be called again since cluster_i is prone to changing if it merges with any of the clusters during the process
          ClusterRankResult clu_rank;
          FeatureCollection clu_comparison;

          if (has_merged)
          {
            clu_rank = RankClusterPair(cluster_i, result.cluster);
            clu_comparison = clu_rank.similarity;
          }
          else
          {
            clu_rank = result.rank_result;
            clu_comparison = result.rank_result.similarity;

          }
          clu_comparison.cluster_density = cluster_i.GetBinFeature();
          clu_comparison.target_density = result.cluster.GetBinFeature();

          var merge_cluster_score = m_clusterjoin_classifier.Score(clu_comparison);

          if (merge_cluster_score > 0)
          {
            cluster_i.JoinCluster(result.cluster);
            merged_clusters.Add(result.cluster.m_record_number);
            m_state.active_clusters.Remove(result.cluster);

            m_cluster_index.clusterIndex.Remove(result.cluster.m_record_number);

            m_cluster_index.InsertClusterState(new OuterClusterState(ClusterDbState.Update, cluster_i));
            m_cluster_index.InsertClusterState(new OuterClusterState(ClusterDbState.Remove, result.cluster));

            foreach (var update_doc in result.cluster.m_document_updates)
            {
              var doc_update_row = new DocumentUpdateRow(cluster_i.m_record_number.ToString(), update_doc);
              m_cluster_index.InsertDocumentState(new DocumentState(ClusterDbState.Update, doc_update_row));

              //m_cluster_index.documentsToUpdate.Add(new DocumentUpdateRow(cluster_i.m_record_number.ToString(), update_doc));
            }

            //the following section propagates the updates to the evaluator

            PutDocumentResult update = new PutDocumentResult();
            update.decision = result.cluster.m_record_number;
            update.relevance_to_cluster = clu_rank.rank + cluster_i.cluster_size_rank_thr;
            update.available_languages = result.cluster.available_languages;
            update.best_clusterindex = -1;

            update.document_updates = result.cluster.m_document_updates;
            update.remove_from_index = true;
            updates.Add(update);


            update = new PutDocumentResult();
            update.decision = cluster_i.m_record_number;
            update.relevance_to_cluster = clu_rank.rank + cluster_i.cluster_size_rank_thr;
            update.available_languages = cluster_i.available_languages;
            update.best_clusterindex = -1;

            update.document_updates = result.cluster.m_document_updates;
            updates.Add(update);

          }

        }
      }

      return updates;
    }


    public List<OuterCluster> ClusterMergingTest()
    {
      List<Ulid> merged_clusters = new List<Ulid>();
      List<OuterCluster> final_clusters = new List<OuterCluster>();

      for (var i = 0; i < m_state.active_clusters.Count; i++)
      {
        var cluster_i = m_state.active_clusters[i];

        if (merged_clusters.Contains(cluster_i.m_record_number))
          continue;

        List<ClusterMergeResult> merge_candidates = new List<ClusterMergeResult>();

        for (var j = i + 1; j < m_state.active_clusters.Count; j++)
        {
          var cluster_j = m_state.active_clusters[j];

          if (merged_clusters.Contains(cluster_j.m_record_number))
            continue;

          var sim_ij = RankClusterPair(cluster_i, cluster_j).similarity;
          var merge_rank = m_merge_classifier.Score(sim_ij);

          if (merge_rank > 0)
          {
            ClusterMergeResult result = new ClusterMergeResult();
            result.score = merge_rank;
            result.cluster = cluster_j;
            merge_candidates.Add(result);
          }
        }

        foreach (var result in merge_candidates.OrderByDescending(cr => cr.score))
        {
          if (m_merge_classifier.Score(RankClusterPair(cluster_i, result.cluster).similarity) > 0) //do this due to the merge updates on cluster_i
          {
            cluster_i.JoinCluster(result.cluster);
            merged_clusters.Add(result.cluster.m_record_number);
          }
        }

        final_clusters.Add(cluster_i);

      }

      return final_clusters;

    }

    // to calc local F1 with candidates as separate clusters, join = false
    // to calc local F1 with candidates as merged clusters, join = true
    // expected procedure is to call this function twice, obtain the LocalF1 values for both possibilities, and then compare them externally
    // changed this from comparing two clusters to comparing a list of clusters; functionally similar, as we can put only i and j, but expanded in case we want to try out a scenario like calc'ing the F1 for all candidate merges
    public double GetLocalF1(List<OuterCluster> candidates, bool join = false)
    {
      var tp = 0;
      var fp = 0;
      var tn = 0;
      var fn = 0;

      List<DocumentUpdate> registered_docs = new List<DocumentUpdate>();
      List<string> labels = new List<string>();
      List<DocumentUpdate> target_docs = new List<DocumentUpdate>();

      foreach (var cluster in candidates)
      {
        foreach (var doc in cluster.m_document_updates)
        {
          target_docs.Add(doc);
          if (!labels.Contains(doc.forced_label))
            labels.Add(doc.forced_label);
        }
      }

      foreach (var label in labels)
      {
        if (!label_index.ContainsKey(label))
          continue;
        var reg_clusters = label_index[label];
        foreach (var c in reg_clusters)
        {
          registered_docs.AddRange(c.m_document_updates);
        }
      }

      foreach (var cluster in candidates)
      {
        foreach (var doc in cluster.m_document_updates)
        {
          foreach (var target_doc in registered_docs)
          {
            if (doc.document_id == target_doc.document_id)
              continue;
            // if the clusters are to be joined, then the final merged cluster would contain all of the target docs
            if (cluster.m_document_updates.Contains(target_doc) || (join && target_docs.Contains(target_doc)))
            {
              if (doc.forced_label == target_doc.forced_label)
                tp += 1;
              else
                fp += 1;
            }
            if (!cluster.m_document_updates.Contains(target_doc))
            {
              if (doc.forced_label == target_doc.forced_label)
                fn += 1;
              else
                tn += 1;
            }
          }
        }
      }

      double p = tp + fp > 0 ? (double)(1 * tp) / (tp + fp) : 0;
      double r = tp + fn > 0 ? (double)(1 * tp) / (tp + fn) : 0;
      double local_f1 = p + r > 0 ? (double)(2 * p * r) / (p + r) : 0;

      return local_f1;
    }

    public double GetAltLocalF1(List<Cluster> candidates)
    {
      var tp = 0;
      var fp = 0;
      //var tn = 0;
      var fn = 0;

      HashSet<string> labels = new HashSet<string>();
      List<OuterCluster> registered_clusters = new List<OuterCluster>();

      foreach (var cluster in candidates)
      {
        foreach (var label in cluster.label_index.Keys)
          labels.Add(label);
      }

      foreach (var label in labels)
      {
        if (!label_index.ContainsKey(label))
          continue;
        var reg_clusters = label_index[label];
        foreach (var c in reg_clusters)
        {
          if (!registered_clusters.Contains(c))
            registered_clusters.Add(c);
        }
      }

      foreach (var cluster in candidates)
      {
        foreach (var label in cluster.label_index.Keys)
        {
          int ndocs_label = cluster.label_index[label];
          int ndocs_other = cluster.m_num_documents - cluster.label_index[label];
          int outside_docs_label = 0;
          //int outside_docs_other = 0;

          foreach (var reg_clu in registered_clusters)
          {
            if (cluster.m_record_number == reg_clu.m_record_number)
              continue;
            if (reg_clu.label_index.ContainsKey(label))
            {
              outside_docs_label += reg_clu.label_index[label];
            }
            //outside_docs_other += reg_clu.m_num_documents - reg_clu.label_index[label];

          }

          tp += ndocs_label * (ndocs_label - 1);
          fp += ndocs_label * ndocs_other;
          //tn += ndocs_label * outside_docs_other;
          fn += ndocs_label * outside_docs_label;

        }
      }

      double p = tp + fp > 0 ? (double)(1 * tp) / (tp + fp) : 0;
      double r = tp + fn > 0 ? (double)(1 * tp) / (tp + fn) : 0;
      double local_f1 = p + r > 0 ? (double)(2 * p * r) / (p + r) : 0;

      return local_f1;

    }



    public double GetAltJoinLocalF1(List<Cluster> candidates)
    {
      var tp = 0;
      var fp = 0;
      //var tn = 0;
      var fn = 0;

      HashSet<string> labels = new HashSet<string>();
      List<OuterCluster> registered_clusters = new List<OuterCluster>();
      Dictionary<string, int> joint_label_index = new Dictionary<string, int>();
      int joint_total_docs = 0;
      HashSet<Ulid> record_numbers = new HashSet<Ulid>();

      foreach (var cluster in candidates)
      {
        foreach (var label in cluster.label_index.Keys)
        {
          labels.Add(label);
          if (!joint_label_index.ContainsKey(label))
            joint_label_index[label] = 0;
          joint_label_index[label] += cluster.label_index[label];
        }
        joint_total_docs += cluster.m_num_documents;
        record_numbers.Add(cluster.m_record_number);
      }

      foreach (var label in labels)
      {
        if (!label_index.ContainsKey(label))
          continue;
        var reg_clusters = label_index[label];
        foreach (var c in reg_clusters)
        {
          if (!registered_clusters.Contains(c))
            registered_clusters.Add(c);
        }
      }

      foreach (var label in joint_label_index.Keys)
      {
        int ndocs_label = joint_label_index[label];
        int ndocs_other = joint_total_docs - joint_label_index[label];
        int outside_docs_label = 0;
        //int outside_docs_other = 0;

        foreach (var reg_clu in registered_clusters)
        {
          if (record_numbers.Contains(reg_clu.m_record_number))
            continue;

          if (reg_clu.label_index.ContainsKey(label))
          {
            outside_docs_label += reg_clu.label_index[label];
          }

          //outside_docs_other += reg_clu.m_num_documents - reg_clu.label_index[label];
        }

        tp += ndocs_label * (ndocs_label - 1);
        fp += ndocs_label * ndocs_other;
        //tn += ndocs_label * outside_docs_other;
        fn += ndocs_label * outside_docs_label;

      }

      double p = tp + fp > 0 ? (double)(1 * tp) / (tp + fp) : 0;
      double r = tp + fn > 0 ? (double)(1 * tp) / (tp + fn) : 0;
      double local_f1 = p + r > 0 ? (double)(2 * p * r) / (p + r) : 0;

      return local_f1;

    }

  }

  public class JsonRepr
  {
    public double[] main_repr;
    public double[] paragraph_repr;
    public double[] title_paragraph_repr;


    public List<double[]> GetReprs()
    {
      return new List<double[]>() { main_repr, paragraph_repr, title_paragraph_repr };
    }
  }

  public class JsonDenseRepr
  {
    public DenseVector main_repr;
    public DenseVector paragraph_repr;
    public DenseVector title_paragraph_repr;

    public JsonDenseRepr()
    {

    }

    public JsonDenseRepr(JsonRepr jsonRepr)
    {
      if (jsonRepr.main_repr != null)
        main_repr = new DenseVector(jsonRepr.main_repr);
      if (jsonRepr.paragraph_repr != null)
        paragraph_repr = new DenseVector(jsonRepr.paragraph_repr);
      if (jsonRepr.title_paragraph_repr != null)
        title_paragraph_repr = new DenseVector(jsonRepr.title_paragraph_repr);
    }

  }

}



