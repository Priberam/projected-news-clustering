using System;

namespace DistilMonoClustering
{
  public class EmbeddingsDocument
  {
    public Guid DocumentId { get; set; }  // Primary and Foreign Key
    public Byte[] TitleEmbeddings { get; set; }
    public Byte[] AverageEmbeddings { get; set; }

    public Document Document { get; set; }
  }
}