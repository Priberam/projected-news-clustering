using System;
using System.Collections.Generic;

namespace DistilMonoClustering
{
  public class Document
  {
    public Guid DocumentId { get; set; }  // Primary Key
    public DateTime? SourceItemDate { get; set; }
    public string ContentDetectedLangCode { get; set; }
    public string SourceItemTitle { get; set; }
    public List<string> SourceItemMainText { get; set; }
    public string SourceItemPageUrl { get; set; }
    public List<string> ContentIptcTopics { get; set; }

    public EmbeddingsDocument Embedding { get; set; }
  }
}