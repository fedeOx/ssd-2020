using System;
using System.Drawing;

namespace SsdWebApi
{
    public class Forecast
    {
        public Forecast()
        {

        }

        public string predictPortfolio(string sourcesNames, int capital, double riskAlpha)
        {
            string res = "";
            string interpreter = @"D:\Users\Federico\Anaconda3\envs\opanalytics\python.exe";
            string environment = "opanalytics";
            int timeout = 10000;
            PythonRunner pyRunner = new PythonRunner(interpreter, environment, timeout);

            try
            {
                string command = $"Models/main.py {sourcesNames} {capital} {riskAlpha}";
                string list = pyRunner.runDosCommands(command);

                if (string.IsNullOrWhiteSpace(list))
                {
                    Console.WriteLine("Error in the script call");
                }
                else
                {
                    string[] lines = list.Split(new[] { Environment.NewLine }, StringSplitOptions.None);
                    foreach (string s in lines)
                    {
                        if (s.StartsWith("BEST_PORTFOLIO"))
                        {
                            res += "\"portfolio\":";
                            var tmp = s.Replace("BEST_PORTFOLIO ", "");
                            res += tmp;
                            res += ",";
                        }

                        if (s.StartsWith("BEST_RETURN"))
                        {
                            res += "\"return\":";
                            var tmp = s.Replace("BEST_RETURN ", "");
                            res += tmp;
                            res += ",";
                        }

                        if (s.StartsWith("BEST_RISK"))
                        {
                            res += "\"risk\":";
                            var tmp = s.Replace("BEST_RISK ", "");
                            res += tmp;
                        }
                    }
                }
            }
            catch (Exception exception)
            {  
                Console.WriteLine(exception.ToString());
                return res;
            }

            return res;
        }
    }
}