using System;
using System.Threading.Tasks;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;
using NLog.Web;
using Microsoft.Extensions.DependencyInjection;
using System.IO;

namespace DistilEvaluation
{
  class Program
  {
    static async Task Main(string[] args)
    {

      var serviceProvider = new ServiceCollection().AddLogging(cfg => cfg.AddConsole())
                                                   .Configure<LoggerFilterOptions>(cfg => cfg.MinLevel = LogLevel.Debug)
                                                   .BuildServiceProvider();
      // get instance of logger
      var logger = serviceProvider.GetService<ILoggerFactory>()
            .CreateLogger<Program>();

      var builder = new ConfigurationBuilder()
        .SetBasePath(Path.GetFullPath(".", Directory.GetCurrentDirectory()))
        .AddJsonFile("appsettings.json", optional: false, reloadOnChange: false);
      //hardcoded; find a better way to do this

      IConfigurationRoot configuration = builder.Build();

      Evaluator evaluator = new Evaluator(configuration, logger);
      await evaluator.Evaluate();
    }
  }
}
