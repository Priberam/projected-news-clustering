
using Microsoft.EntityFrameworkCore;
using System;
using System.ComponentModel.DataAnnotations;
using System.IO;

namespace DistilMonoClustering
{
  public class NearestClusterDbContext : DbContext
  {
    public DbSet<ClusterRow> ClusterRows { get; set; }
    public DbSet<PoolState> PoolState { get; set; }
    public DbSet<DocumentUpdateRow> DocumentUpdateRows { get; set; }

    public NearestClusterDbContext(DbContextOptions<NearestClusterDbContext> options)
        : base(options)
    {
    }
  }
  public class ClusterRow
  {
    [Key]
    public string ClusterId { get; set; }
    public Byte[] Centroid { get; set; }
    public Byte[] TitleCentroid { get; set; }
    public Byte[] ParagraphCentroid { get; set; }
    public Byte[] TitleParagraphCentroid { get; set; }
    public DateTime NewestTimestamp { get; set; }
    public DateTime OldestTimestamp { get; set; }
    public int NumDocuments { get; set; }
    public int CompleteNumDocuments { get; set; }
    public double SumTimestamp { get; set; }
    public double SumSqTimestamp { get; set; }
    public double SumSqAcceptanceScores { get; set; }
    public double SumRelevance { get; set; }

    public ClusterRow()
    {

    }

    public ClusterRow(Cluster cluster)
    {
      ClusterId = cluster.m_record_number.ToString();
      NewestTimestamp = cluster.m_newest_timestamp;
      OldestTimestamp = cluster.m_oldest_timestamp;
      NumDocuments = cluster.m_num_documents;
      CompleteNumDocuments = cluster.m_complete_num_documents;
      SumTimestamp = cluster.m_sum_timestamp;
      SumSqTimestamp = cluster.m_sumsq_timestamp;
      SumSqAcceptanceScores = cluster.m_sumsq_acceptancescores;
      SumRelevance = cluster.m_sum_relevance;

      Centroid = GetByteRepresentation(cluster.m_centroid);
      TitleCentroid = GetByteRepresentation(cluster.title_centroid);
      ParagraphCentroid = GetByteRepresentation(cluster.paragraph_centroid);
      TitleParagraphCentroid = GetByteRepresentation(cluster.title_paragraph_centroid);
    }


    public Byte[] GetByteRepresentation(DenseVector denseVector)
    {
      MemoryStream mem_stream = new MemoryStream();
      BinaryWriter binary_writer = new BinaryWriter(mem_stream);
      foreach (var el in denseVector.dense_vector)
      {
        binary_writer.Write(el);
      }
      binary_writer.Flush();

      Byte[] byte_repr = mem_stream.ToArray();

      return byte_repr;

    }

    public void UpdateRow(Cluster cluster)
    {
      ClusterId = cluster.m_record_number.ToString();
      NewestTimestamp = cluster.m_newest_timestamp;
      OldestTimestamp = cluster.m_oldest_timestamp;
      NumDocuments = cluster.m_num_documents;
      CompleteNumDocuments = cluster.m_complete_num_documents;
      SumTimestamp = cluster.m_sum_timestamp;
      SumSqTimestamp = cluster.m_sumsq_timestamp;
      SumSqAcceptanceScores = cluster.m_sumsq_acceptancescores;
      SumRelevance = cluster.m_sum_relevance;

      Centroid = GetByteRepresentation(cluster.m_centroid);
      TitleCentroid = GetByteRepresentation(cluster.title_centroid);
      ParagraphCentroid = GetByteRepresentation(cluster.paragraph_centroid);
      TitleParagraphCentroid = GetByteRepresentation(cluster.title_paragraph_centroid);

    }

  }

  public class PoolState
  {
    [Key]
    public string PoolId { get; set; }
    public DateTime LastUpdateDatetime { get; set; }

    public PoolState()
    {

    }
    public PoolState(string scenarioId, DateTime timestamp)
    {
      PoolId = scenarioId;
      LastUpdateDatetime = timestamp;
    }


  }


  public class DocumentUpdateRow
  {
    [Key]
    public string DocumentId { get; set; }
    public string ClusterId { get; set; }
    public string Language { get; set; }
    public string GroupId { get; set; }
    public string ForcedLabel { get; set; }
    public double Similarity { get; set; }
    public DateTime Timestamp { get; set; }
    public DateTime UpdateTimestamp { get; set; }
    public DocumentUpdateRow()
    {

    }

    public DocumentUpdateRow(string cluster_id, DocumentUpdate document_update)
    {
      ClusterId = cluster_id;
      DocumentId = document_update.document_id;
      Language = document_update.language;
      GroupId = document_update.group_id;
      ForcedLabel = document_update.forced_label;
      Similarity = document_update.similarity;
      Timestamp = document_update.timestamp;
      UpdateTimestamp = document_update.update_timestamp;
    }

    public void UpdateRow(DocumentUpdateRow document_update_row)
    {
      ClusterId = document_update_row.ClusterId;
      DocumentId = document_update_row.DocumentId;
      Language = document_update_row.Language;
      GroupId = document_update_row.GroupId;
      ForcedLabel = document_update_row.ForcedLabel;
      Similarity = document_update_row.Similarity;
      Timestamp = document_update_row.Timestamp;
      UpdateTimestamp = document_update_row.UpdateTimestamp;
    }
  };
}