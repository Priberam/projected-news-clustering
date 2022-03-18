using System;
using System.Collections.Generic;
using System.Linq;


namespace DistilMonoClustering
{
  public class DumpDocument
  {
    public Guid Id { get; set; }
    public DateTime? SourceItemDate { get; set; }
    public string ContentDetectedLangCode { get; set; }
    public string SourceItemTitle { get; set; }
    public string SourceItemMainText { get; set; }
    public string SourceItemPageUrl { get; set; }
    public List<String> ContentIptcTopics { get; set; }
    public CustomMetadata CustomMetadata { get; set; }

    public DumpDocument()
    {
      ContentIptcTopics = new List<string>();
    }

    private static readonly List<string> newlineTags = new List<string>
    {
        "address",
        "article",
        "aside",
        "blockquote",
        "canvas",
        "dd",
        "div",
        "dl",
        "dt",
        "fieldset",
        "figcaption",
        "figure",
        "footer",
        "form",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "hr",
        "li",
        "main",
        "nav",
        "noscript",
        "ol",
        "p",
        "pre",
        "section",
        "table",
        "tfoot",
        "ul",
        "video"
    };
    private static readonly List<char> spaces = new List<char> { ' ', '\t' };

    class BlockEnd
    {
      public int Index;
      public bool IsNewLine;
    }

    public Document Extract()
    {
      var doc = new Document
      {
        DocumentId = Id,
        SourceItemDate = SourceItemDate,
        ContentDetectedLangCode = ContentDetectedLangCode,
        SourceItemTitle = SourceItemTitle,
        SourceItemMainText = new List<string>(),
        SourceItemPageUrl = SourceItemPageUrl,
        ContentIptcTopics = new List<string>(ContentIptcTopics)
      };

      // Get the ordered ending indexes of the formatting blocks
      var blocksEnd = new List<BlockEnd>();
      var textFormat = CustomMetadata?.OriginalFormatting?.SourceItemMainText;

      if (textFormat != null)
      {
        foreach (KeyValuePair<string, List<FormatBlock>> tag in textFormat)
        {
          bool isNewLine = newlineTags.Contains(tag.Key);
          blocksEnd.AddRange(
            tag.Value.Select(b => new BlockEnd
            {
              Index = b.End,
              IsNewLine = isNewLine
            }).ToList());
        }
        blocksEnd = blocksEnd.OrderBy(e => e.Index).ToList();
      }

      // Remove extra spaces and split paragraphs (using the html blocks endings)
      string par;
      var curParagraph = new List<char>();
      bool ignoreSpace = false;
      for (var idx = 0; idx < SourceItemMainText.Length; idx++)
      {
        char c = SourceItemMainText[idx];
        bool isSpace = spaces.Contains(c);
        if (blocksEnd.Any() && idx == blocksEnd.ElementAt(0).Index)
        {
          // Check if the html block(s) mark the end of a paragraph
          bool isBlock = false;
          while (blocksEnd.Any() && idx == blocksEnd.ElementAt(0).Index)
          {
            isBlock |= blocksEnd.ElementAt(0).IsNewLine;
            blocksEnd.RemoveAt(0);
          }
          if (isBlock)
          {
            // Add the paragraph if it has any content
            par = new string(curParagraph.ToArray()).Trim();
            curParagraph.Clear();
            if (par.Length > 0)
            {
              doc.SourceItemMainText.Add(par);
            }
          }
          else if (c != ' ')
          {
            // Make sure there is a space after this (style?) tag
            curParagraph.Add(' ');
          }
        }

        // Store the character unless it's a newline or an extra space
        if (c == '\n')
        {
          ignoreSpace = true;
        }
        else if (!isSpace || !ignoreSpace)
        {
          ignoreSpace = isSpace;
          curParagraph.Add(c);
        }
      }
      // Add the last paragraph (if any)
      par = new string(curParagraph.ToArray()).Trim();
      if (par.Length > 0)
      {
        doc.SourceItemMainText.Add(par);
      }

      return doc;
    }
  }

  public class CustomMetadata
  {
    public OriginalFormatting OriginalFormatting { get; set; }
  }

  public class OriginalFormatting
  {
    public Dictionary<string, List<FormatBlock>> SourceItemMainText { get; set; }

    public OriginalFormatting()
    {
      SourceItemMainText = new Dictionary<string, List<FormatBlock>>();
    }
  }

  public class FormatBlock
  {
    public int Start { get; set; }
    public int End { get; set; }
    public int Ord { get; set; }
  }
}