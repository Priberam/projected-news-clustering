using DistilMonoClustering;
using System.Collections.Generic;
using System.Threading.Tasks;
using System.IO;
using System;
using Newtonsoft.Json;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;
using System.Linq;
using ScottPlot;

namespace DistilEvaluation
{
  public struct CooccurrenceMatrix
  {
    public Dictionary<string, int> true_label_map;
    public Dictionary<string, int> hyp_label_map;
    public int[][] cooccurrence_matrix;

  }

  public interface IEvaluator
  {
    Task Evaluate();
  }

  public class Evaluator : IEvaluator
  {
    private IConfigurationRoot configuration;
    private ILogger logger;
    string gold_dataset_path;
    string cross_dataset_path;
    string cross_only_path;
    bool options_build_cross_eval = true;
    bool UseUniqueCrosslingualModule = true;
    List<double> num_clus_stat_multi = new List<double>();
    Dictionary<string, List<double>> num_clus_stat = new Dictionary<string, List<double>>();
    Dictionary<string, StreamClustering> modules = new Dictionary<string, StreamClustering>();

    public Evaluator()
    {
      
    }


    public Evaluator(IConfigurationRoot configuration, ILogger logger)
    {
      this.configuration = configuration;
      this.logger = logger;
      this.gold_dataset_path = configuration[$"TrainEval:InputDatasetPath"];
      this.cross_dataset_path = configuration[$"TrainEval:CrossDatasetPath"];
      //this.cross_only_path = configuration[$"TrainEval:CrossOnlyPath"];
    }

    public async Task Evaluate()
    {
      List<DatasetSample> samples = JsonConvert.DeserializeObject<List<DatasetSample>>(File.ReadAllText(this.gold_dataset_path));
      Dictionary<string, Dictionary<string, bool>> cross_dataset = ReadCrossDataset(this.cross_dataset_path);
      //Dictionary<string, Dictionary<string, bool>> cross_only = ReadCrossDataset(this.cross_only_path);

      Dictionary<string, Dictionary<string, HashSet<string>>> mono_cluster_index = new Dictionary<string, Dictionary<string, HashSet<string>>>();
      Dictionary<string, Dictionary<string, List<string>>> cross_cluster_index = new Dictionary<string, Dictionary<string, List<string>>>();
      List<int> best_cluster_indexes = new List<int>();

      HashSet<string> registered_document_ids = new HashSet<string>();


      int doc_i = 0;
      int step = 1000;


      foreach (DatasetSample sample in samples)
      {
        //ClusteringUpdate update = new ClusteringUpdate();
        List<ClusteringUpdate> updates = new List<ClusteringUpdate>();

        if ((doc_i != 0 && doc_i % step == 0) || doc_i == (samples.Count - 1))
        {
          if (!bool.Parse(configuration[$"TrainEval:UseForcedLabel"]))
          {
            EvaluateRoutine(mono_cluster_index, samples);
            if (options_build_cross_eval)
            {
              EvaluateCrosslingual(cross_cluster_index, cross_dataset, registered_document_ids);

              //Console.WriteLine("Crosslingual connections only: ");
              //EvaluateCrosslingual(cross_cluster_index, cross_only, registered_document_ids, false);
              //SaveTopplingPlot(modules["multi"]);
            }
          }
          if (bool.Parse(configuration[$"TrainEval:EvaluateClusterRecall"]))
            EvaluateClusterRecall(best_cluster_indexes, doc_i);
          //SaveNumClustersPlot();
          //SaveClusterDensityPlot(modules["multi"].m_state.active_clusters, modules["multi"].boot_time);
          Console.WriteLine("Progress: " + Math.Round(Convert.ToDecimal((doc_i * 100) / samples.Count), 2) + "%");
        }

        //if (sample.lang != "spa" && sample.lang != "eng" && sample.lang != "deu")
        //{
        //  doc_i += 1;
        //  continue;
        //}

        var module = sample.lang;

        if (UseUniqueCrosslingualModule)
        {
          module = "multi";
        }
        if (!modules.ContainsKey(module))
        {
          modules[module] = new StreamClustering(configuration, logger);    //update DistilEvaluation's appsettings to include module name
        }

        var arbitrary_skip = false;

        if (bool.Parse(configuration[$"TrainEval:DeviateFromGoldSamples"]))
          arbitrary_skip = doc_i % int.Parse(configuration[$"TrainEval:DeviationStep"]) == 0 ? true : false;

        //update = await modules[module].PutDocument(sample.ConvertToDocumentPayload(bool.Parse(configuration[$"TrainEval:UseForcedLabel"]), arbitrary_skip));
        //var watch_put = System.Diagnostics.Stopwatch.StartNew();
        updates = await modules[module].PutDocument(sample.ConvertToDocumentPayload(bool.Parse(configuration[$"TrainEval:UseForcedLabel"]), arbitrary_skip));

        //watch_put.Stop();
        //Console.WriteLine("PutDocument(): " + watch_put.ElapsedMilliseconds + " ms");
        //Console.WriteLine();


        foreach (var update in updates)
        {
          foreach (var doc_update in update.document_updates)
          {
            if (update.remove_from_index)
            {
              mono_cluster_index[doc_update.group_id][update.cluster_id].Remove(doc_update.document_id);
              cross_cluster_index["multi"][update.cluster_id].Remove(doc_update.document_id);
              if (mono_cluster_index[doc_update.group_id][update.cluster_id].Count == 0)
                mono_cluster_index[doc_update.group_id].Remove(update.cluster_id);
              if (cross_cluster_index["multi"][update.cluster_id].Count == 0)
                cross_cluster_index["multi"].Remove(update.cluster_id);
              registered_document_ids.Remove(doc_update.document_id);
              continue;
            }

            if (update.type == "mono")
            {
              if (!mono_cluster_index.ContainsKey(doc_update.group_id))
                mono_cluster_index[doc_update.group_id] = new Dictionary<string, HashSet<string>>();

              if (!mono_cluster_index[doc_update.group_id].ContainsKey(update.cluster_id))
                mono_cluster_index[doc_update.group_id][update.cluster_id] = new HashSet<string>();

              mono_cluster_index[doc_update.group_id][update.cluster_id].Add(doc_update.document_id);


              if (bool.Parse(configuration[$"Main:GenerateDumpFile"]))
              {
                //string dump_name = configuration[$"Main:DumpFilePath"] + "\\multi_" + modules["multi"].boot_time + ".clu.dump";
                string dump_name = configuration[$"Main:DumpFilePath"] + "\\" + doc_update.group_id + "_" + modules["multi"].boot_time + ".clu.dump.jsonl";

                using (StreamWriter clusters_file = File.AppendText(dump_name))
                {
                  //clusters_file.WriteLine(doc_update.document_id + "\t" + update.cluster_id);
                  clusters_file.WriteLine("{\"documentId\": \"" + doc_update.document_id + "\", \"clusterId\": \"" + update.cluster_id + "\", \"similarity\": \"" + doc_update.similarity.ToString() + "\", \"availableLanguages\": [\"" + String.Join("\", \"", update.available_languages) + "\"]}");
                }

              }

            }

            if (options_build_cross_eval)
            {
              if (!cross_cluster_index.ContainsKey("multi"))
                cross_cluster_index["multi"] = new Dictionary<string, List<string>>();

              if (!cross_cluster_index["multi"].ContainsKey(update.cluster_id))
                cross_cluster_index["multi"][update.cluster_id] = new List<string>();

              cross_cluster_index["multi"][update.cluster_id].Add(doc_update.document_id);
              registered_document_ids.Add(doc_update.document_id);

              best_cluster_indexes.Add(update.best_clusterindex);

              if (bool.Parse(configuration[$"Main:GenerateDumpFile"]))
              {
                //string dump_name = configuration[$"Main:DumpFilePath"] + "\\multi_" + modules["multi"].boot_time + ".clu.dump";
                string dump_name = configuration[$"Main:DumpFilePath"] + "\\multi_" + modules["multi"].boot_time + ".clu.dump.jsonl";

                using (StreamWriter clusters_file = File.AppendText(dump_name))
                {
                  //clusters_file.WriteLine(doc_update.document_id + "\t" + update.cluster_id);
                  clusters_file.WriteLine("{\"documentId\": \"" + doc_update.document_id + "\", \"clusterId\": \"" + update.cluster_id + "\", \"similarity\": \"" + doc_update.similarity.ToString() + "\", \"availableLanguages\": [\"" + String.Join("\", \"", update.available_languages) + "\"]}");
                }

              }

            }
          }

        }

        //temp to debug
        List<string> group_ids = new List<string>(new string[] { "clusters_eng", "clusters_spa", "clusters_deu" });
        foreach (var m in group_ids)
        {
          if (!num_clus_stat.ContainsKey(m))
            num_clus_stat[m] = new List<double>();
          if (!mono_cluster_index.ContainsKey(m))
            num_clus_stat[m].Add(0);
          else
            num_clus_stat[m].Add(mono_cluster_index[m].Count);
        }
        num_clus_stat_multi.Add(cross_cluster_index["multi"].Count);

        doc_i += 1;

      }

      //ClusterMergingTest(cross_dataset, registered_document_ids);
    }

    public Dictionary<string, Dictionary<string, bool>> ReadCrossDataset(string dataset_path)
    {
      Dictionary<string, Dictionary<string, bool>> cross_dataset = new Dictionary<string, Dictionary<string, bool>>();

      using (StreamReader dataset_reader = new StreamReader(dataset_path))
      {
        string line;
        while ((line = dataset_reader.ReadLine()) != null)
        {
          var line_parts = line.Split("\t");
          bool connection = line_parts[2] == "positive" ? true : false;
          if (!cross_dataset.ContainsKey(line_parts[0]))
            cross_dataset[line_parts[0]] = new Dictionary<string, bool>();
          cross_dataset[line_parts[0]][line_parts[1]] = connection;
        }
      }
      return cross_dataset;
    }

    public void EvaluateClusterRecall(List<int> index_array, int iterated_docs)
    {
      double highest_index = index_array.Max();
      double recall_1 = index_array.Count(x => x < 1);
      double recall_5 = index_array.Count(x => x < 5);
      double recall_10 = index_array.Count(x => x < 10);
      double recall_20 = index_array.Count(x => x < 20);
      double recall_50 = index_array.Count(x => x < 50);
      double recall_100 = index_array.Count(x => x < 100);
      double recall_highest = index_array.Count(x => x <= highest_index); //sanity check; should be 1

      Console.WriteLine("Highest cluster index rank: " + highest_index);
      Console.WriteLine("Cluster Recall@1 = " + recall_1 / iterated_docs);
      Console.WriteLine("Cluster Recall@5 = " + recall_5 / iterated_docs);
      Console.WriteLine("Cluster Recall@10 = " + recall_10 / iterated_docs);
      Console.WriteLine("Cluster Recall@20 = " + recall_20 / iterated_docs);
      Console.WriteLine("Cluster Recall@50 = " + recall_50 / iterated_docs);
      Console.WriteLine("Cluster Recall@100 = " + recall_100 / iterated_docs);
      Console.WriteLine("Cluster Recall@" + highest_index + " (highest) = " + recall_highest / iterated_docs);

    }


    public void EvaluateCrosslingual(Dictionary<string, Dictionary<string, List<string>>> cross_cluster_index, Dictionary<string, Dictionary<string, bool>> cross_dataset, HashSet<string> registered_document_ids, bool all_connections = true)
    {
      //not taking into account things outside the evaluated samples

      var tp = 0;
      var fp = 0;
      var tn = 0;
      var fn = 0;

      foreach (var cluster_id in cross_cluster_index["multi"].Keys)
      {
        var source_cluster = cross_cluster_index["multi"][cluster_id];

        for (var x = 0; x < source_cluster.Count; x++)
        {
          if (cross_dataset.ContainsKey(source_cluster[x]))
          {
            var source_doc_id = source_cluster[x];
            HashSet<string> docs_to_eval = new HashSet<string>();
            docs_to_eval.UnionWith(source_cluster);
            docs_to_eval.UnionWith(cross_dataset[source_doc_id].Keys);
            docs_to_eval.IntersectWith(registered_document_ids);


            foreach (var target_doc_id in docs_to_eval)
            {
              if (target_doc_id == source_doc_id)
                continue;
              if (source_cluster.Contains(target_doc_id))
              {
                if (cross_dataset[source_doc_id].ContainsKey(target_doc_id) && cross_dataset[source_cluster[x]][target_doc_id] == true)
                  tp += 1;
                if ((cross_dataset[source_doc_id].ContainsKey(target_doc_id) && cross_dataset[source_cluster[x]][target_doc_id] == false) ||
                (!cross_dataset[source_doc_id].ContainsKey(target_doc_id) && all_connections))
                  fp += 1;
              }
              if (!source_cluster.Contains(target_doc_id))
              {
                if (cross_dataset[source_doc_id].ContainsKey(target_doc_id) && cross_dataset[source_cluster[x]][target_doc_id] == true)
                  fn += 1;
                if ((cross_dataset[source_doc_id].ContainsKey(target_doc_id) && cross_dataset[source_cluster[x]][target_doc_id] == false))
                  tn += 1;
              }
            }
          }
        }
      }

      tp /= 2;
      fp /= 2;
      tn /= 2;
      fn /= 2;

      Console.WriteLine("Crosslingual module:");

      string s = "";

      s += "TP: " + tp + ", FP: " + fp + ", TN: " + tn + ", FN: " + fn + ", ";

      double acc = (tp + tn + fp + fn > 0) ? (double)(1 * (tp + tn)) / (tp + tn + fp + fn) : 0;
      double p = tp + fp > 0 ? (double)(1 * tp) / (tp + fp) : 0;
      double r = tp + fn > 0 ? (double)(1 * tp) / (tp + fn) : 0;
      double f1 = p + r > 0 ? (double)(2 * p * r) / (p + r) : 0;

      s += "P: " + p + ", R: " + r + ", F1: " + f1 + ", ACC: " + acc + ", N_CLU: " + cross_cluster_index["multi"].Count + "\n";

      Console.WriteLine(s);

    }


    public void SaveNumClustersPlot()
    {
      var plt = new ScottPlot.Plot(600, 400);
      plt.AxisBounds(0, int.MaxValue, 0, int.MaxValue);

      double[] xs = DataGen.Consecutive(num_clus_stat_multi.Count);
      double[] num_clusters_multi = num_clus_stat_multi.ToArray();


      foreach (var num_cl in num_clus_stat.Keys)
      {
        double[] num_clusters = num_clus_stat[num_cl].ToArray();
        plt.PlotScatter(xs, num_clusters, label: num_cl);
      }

      plt.PlotScatter(xs, num_clusters_multi, label: "multi");

      plt.Legend();

      plt.Title("Number of Clusters Over Time");
      plt.YLabel("Num. Clusters");
      plt.XLabel("Num. Processed Documents");

      plt.SaveFig(configuration[$"Main:DumpFilePath"] + "\\multi_" + modules["multi"].boot_time + "_" + num_clus_stat_multi.Count.ToString() + "_nclusters.png");
    }

    public void SaveClusterDensityPlot(List<OuterCluster> clusters, string boot_time)
    {
      var plt = new ScottPlot.Plot(600, 400);
      string[] labels = { "1", "2", "3-5", "6-10", "11-25", "26-50", "51-100", "101-200", "201-500", "500+" };
      double[] xs = DataGen.Consecutive(labels.Length);

      double[] bin_bounds = { 1, 2, 5, 10, 25, 50, 100, 200, 500 };

      double[] density = GetClusterDensityVector(bin_bounds, clusters);

      plt.PlotBar(xs, density, showValues: true);

      plt.Grid(enableVertical: false, lineStyle: LineStyle.Dot);

      plt.Title("Cluster Density (Num. Docs)");
      plt.YLabel("Num. Docs in Cluster");
      plt.XLabel("Total Num. Clusters");

      // apply custom axis tick labels
      plt.XTicks(xs, labels);

      plt.SaveFig(configuration[$"Main:DumpFilePath"] + "\\multi_" + boot_time + "_" + num_clus_stat_multi.Count.ToString() + "_density.png");

    }

    public void SaveTopplingPlot(StreamClustering module)
    {
      var plt = new ScottPlot.Plot(600, 400);
      double[] xs = DataGen.Consecutive(module.toppling_occurrences.Count);
      double[] accepted_toppling = new double[module.toppling_occurrences.Count];
      double[] refused_on_merge = new double[module.toppling_occurrences.Count];
      //double[] refused_on_rank = new double[module.toppling_occurrences.Count];
      //double[] refused_on_micro = new double[module.toppling_occurrences.Count];
      var at_count = 0;
      var rom_count = 0;
      //var ror_count = 0;
      //var romi_count = 0;

      for (int v = 0; v < module.toppling_occurrences.Count; v += 1)
      {
        if (module.toppling_occurrences[v] == 1)
          at_count += 1;
        if (module.toppling_occurrences[v] == -1)
          rom_count += 1;
        //if (module.toppling_occurrences[v] == -2)
        //  ror_count += 1;
        //if (module.toppling_occurrences[v] == 0)
        //  romi_count += 1;

        accepted_toppling[v] = at_count;
        refused_on_merge[v] = rom_count;
        //refused_on_rank[v] = ror_count;
        //refused_on_micro[v] = romi_count;
      }

      plt.PlotScatter(xs, accepted_toppling, label: "accepted");
      //plt.PlotScatter(xs, refused_on_micro, label: "refused on micro");
      //plt.PlotScatter(xs, refused_on_rank, label: "refused on rank");
      plt.PlotScatter(xs, refused_on_merge, label: "refused on merge");

      plt.Legend(location: legendLocation.upperLeft);

      plt.Title("Domino Toppling Attempt Count");
      plt.YLabel("Num. Micro Clusters");
      plt.XLabel("Num. Processed Documents");

      plt.SaveFig(configuration[$"Main:DumpFilePath"] + "\\multi_" + module.boot_time + "_" + num_clus_stat_multi.Count.ToString() + "_toppling.png");
    }


    public void ClusterMergingTest(Dictionary<string, Dictionary<string, bool>> cross_dataset, HashSet<string> registered_document_ids)
    {
      StreamClustering module_replica = modules["multi"];
      List<OuterCluster> final_clusters = module_replica.ClusterMergingTest();

      Dictionary<string, List<string>> cluster_index = new Dictionary<string, List<string>>();

      foreach (var cluster in final_clusters)
      {
        if (!cluster_index.ContainsKey(cluster.m_record_number.ToString()))
        {
          cluster_index[cluster.m_record_number.ToString()] = new List<string>();
        }

        foreach (var doc in cluster.m_document_updates)
        {
          cluster_index[cluster.m_record_number.ToString()].Add(doc.document_id);
        }
      }

      var cross_cluster_index = new Dictionary<string, Dictionary<string, List<string>>>();
      cross_cluster_index["multi"] = cluster_index;

      SaveClusterDensityPlot(final_clusters, module_replica.boot_time + "_merged");
      EvaluateCrosslingual(cross_cluster_index, cross_dataset, registered_document_ids);

    }


    public void EvaluateRoutine(Dictionary<string, Dictionary<string, HashSet<string>>> mono_cluster_index, List<DatasetSample> gold_samples)
    {
      List<string> langs = new List<string>();

      Dictionary<string, Dictionary<string, string>> clusters_pred = new Dictionary<string, Dictionary<string, string>>();
      Dictionary<string, List<string>> clusters_to_docs_pred = new Dictionary<string, List<string>>();

      Dictionary<string, Dictionary<string, string>> clusters_gold = new Dictionary<string, Dictionary<string, string>>();
      Dictionary<string, List<string>> clusters_to_docs_gold = new Dictionary<string, List<string>>();


      var ts = DateTime.Now.Ticks.ToString();
      foreach (var item in mono_cluster_index)
      {
        string cluster_group = item.Key;
        if (!langs.Contains(cluster_group))
          langs.Add(cluster_group);

        var clusters = item.Value;
        string dump_name = cluster_group + "_" + ts + ".clu.dump";

        //using (StreamWriter clusters_file = new StreamWriter(dump_name))
        //{
        foreach (var clu in clusters)
        {
          string cluster = clu.Key;
          var documents = clu.Value;
          foreach (string doc in documents)
          {
            if (!clusters_pred.ContainsKey(cluster_group))
              clusters_pred[cluster_group] = new Dictionary<string, string>();
            clusters_pred[cluster_group][doc] = cluster;

            if (!clusters_to_docs_pred.ContainsKey(cluster))
              clusters_to_docs_pred[cluster] = new List<string>();
            clusters_to_docs_pred[cluster].Add(doc);


            //clusters_file.WriteLine(doc + "\t" + cluster);
          }
        }
        //}
      }

      //using (StreamReader dump_file = new StreamReader(dump_name))

      foreach (var gold_doc in gold_samples)
      {
        //hotfix; set the allowed languages in the settings later
        if ((!clusters_pred.ContainsKey("clusters_" + gold_doc.lang)))
          continue;

        if (!clusters_pred["clusters_" + gold_doc.lang].ContainsKey(gold_doc.id))
          continue;

        if (!clusters_gold.ContainsKey("clusters_" + gold_doc.lang))
          clusters_gold["clusters_" + gold_doc.lang] = new Dictionary<string, string>();

        clusters_gold["clusters_" + gold_doc.lang][gold_doc.id] = gold_doc.cluster;
        if (!clusters_to_docs_gold.ContainsKey(gold_doc.cluster))
          clusters_to_docs_gold[gold_doc.cluster] = new List<string>();

        clusters_to_docs_gold[gold_doc.cluster].Add(gold_doc.id);
      }

      foreach (var lang in clusters_pred.Keys)
      {
        Dictionary<string, int> id_to_index = new Dictionary<string, int>();
        var index = -1;
        foreach (var key in clusters_pred[lang].Keys)
        {
          index += 1;
          id_to_index[key] = index;
        }

        List<string> true_labels = new List<string>(new string[clusters_gold[lang].Count]);
        List<string> pred_labels = new List<string>(new string[clusters_pred[lang].Count]);

        foreach (var gold_key in clusters_gold[lang].Keys)
          true_labels[id_to_index[gold_key]] = clusters_gold[lang][gold_key];

        foreach (var pred_key in clusters_pred[lang].Keys)
          pred_labels[id_to_index[pred_key]] = clusters_pred[lang][pred_key];

        Console.WriteLine("Module " + lang + ":");
        ScoreSet(true_labels, pred_labels, mono_cluster_index[lang]);
      }
    }



    public double[] GetClusterDensityVector(double[] bin_bounds, List<OuterCluster> clusters)
    {
      double[] density = new double[bin_bounds.Length + 1];

      for (var i = 0; i < clusters.Count; i++)
      {
        for (var j = 0; j < bin_bounds.Length; j++)
        {
          if (clusters[i].m_num_documents <= bin_bounds[j])
          {
            density[j] += 1;
            break;
          }
          if (clusters[i].m_num_documents > bin_bounds[bin_bounds.Length - 1])
          {
            density[density.Length - 1] += 1;
            break;
          }

        }
      }

      return density;

    }

    public void ScoreSet(List<string> true_labels, List<string> pred_labels, Dictionary<string, HashSet<string>> mono_cluster)
    {
      CooccurrenceMatrix c_matrix = GetCooccurrenceMatrix(true_labels, pred_labels);
      List<int> results = GetTpFpTnFn(c_matrix.cooccurrence_matrix);

      int tp = results[0];
      int fp = results[1];
      int tn = results[2];
      int fn = results[3];

      string s = "";

      s += "TP: " + tp + ", FP: " + fp + ", TN: " + tn + ", FN: " + fn + ", ";

      double acc = (tp + tn + fp + fn > 0) ? (double)(1 * (tp + tn)) / (tp + tn + fp + fn) : 0;
      double p = tp + fp > 0 ? (double)(1 * tp) / (tp + fp) : 0;
      double r = tp + fn > 0 ? (double)(1 * tp) / (tp + fn) : 0;
      double f1 = p + r > 0 ? (double)(2 * p * r) / (p + r) : 0;

      s += "P: " + p + ", R: " + r + ", F1: " + f1 + ", ACC: " + acc + ", N_CLU: " + mono_cluster.Count + "\n";

      Console.WriteLine(s);

    }



    public CooccurrenceMatrix GetCooccurrenceMatrix(List<string> true_labels, List<string> pred_labels)
    {
      CooccurrenceMatrix c_matrix = new CooccurrenceMatrix();
      c_matrix.true_label_map = new Dictionary<string, int>();
      c_matrix.hyp_label_map = new Dictionary<string, int>();

      int i = 0;
      foreach (var l in true_labels)
      {
        if (!c_matrix.true_label_map.ContainsKey(l))
        {
          c_matrix.true_label_map[l] = i;
          i += 1;
        }
      }

      i = 0;
      foreach (var l in pred_labels)
      {
        if (!c_matrix.hyp_label_map.ContainsKey(l))
        {
          c_matrix.hyp_label_map[l] = i;
          i += 1;
        }
      }

      c_matrix.cooccurrence_matrix = new int[c_matrix.true_label_map.Count][];
      for (i = 0; i < c_matrix.true_label_map.Count; i++)
      {
        c_matrix.cooccurrence_matrix[i] = new int[c_matrix.hyp_label_map.Count];
      }

      for (i = 0; i < true_labels.Count; i++)
      {
        c_matrix.cooccurrence_matrix[c_matrix.true_label_map[true_labels[i]]][c_matrix.hyp_label_map[pred_labels[i]]] += 1;
      }

      return c_matrix;
    }

    public List<int> GetTpFpTnFn(int[][] cooccurrence_matrix)
    {
      List<int> result = new List<int>();

      var row_sum = new int[cooccurrence_matrix.Length];
      var col_sum = new int[cooccurrence_matrix[0].Length];
      var total_cm = 0;
      var tp = 0;
      var tp_plus_fp = 0;
      var tp_plus_fn = 0;

      for (var i = 0; i < cooccurrence_matrix.Length; i++)
      {
        for (var j = 0; j < cooccurrence_matrix[i].Length; j++)
        {
          row_sum[i] += cooccurrence_matrix[i][j];
          col_sum[j] += cooccurrence_matrix[i][j];
          tp += Combination(cooccurrence_matrix[i][j], 2);    //not sure if this is the right way; check the logic more carefully
          total_cm += cooccurrence_matrix[i][j];

        }
      }


      for (int i = 0; i < row_sum.Length; i++)
      {
        tp_plus_fn += Combination(row_sum[i], 2);
      }

      for (int i = 0; i < col_sum.Length; i++)
      {
        tp_plus_fp += Combination(col_sum[i], 2);

      }

      var fp = tp_plus_fp - tp;
      var fn = tp_plus_fn - tp;
      var tn = Combination(total_cm, 2) - tp - fp - fn;

      result.Add(tp);
      result.Add(fp);
      result.Add(tn);
      result.Add(fn);

      return result;

    }

    public static int Combination(long n, long k)
    {
      if (n == 0)
        return 0;

      double sum = 0;
      for (int i = 0; i < k; i++)
      {
        sum += Math.Log10(n - i);
        sum -= Math.Log10(i + 1);
      }
      return (int)Math.Pow(10, sum);
    }

  }
}