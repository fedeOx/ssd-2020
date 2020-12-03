using System;
using System.Drawing;

namespace SsdWebApi
{
    public class Forecast
    {
        public Forecast()
        {

        }

        /*
            Produce in output una stringa costituita da due campi:
            - "text": "parte testuale da visualizzare sull'interfaccia testuale del browser";
            - "img": "codifica base 64 dell'immagine di eventuali grafici".
            Tale stringa viene spedida a chi invoca questo metodo ("IndiciController") che provvede ad aggiungere "{" e "}" alla stringa
            creando quindi un JSON a tutti gli effetti, che sarà quello che viene restituito al browser.
        */
        public string forecastSARIMAindex(string attribute)
        {
            string res = "\"text\":\"";
            string interpreter = @"D:\Users\Federico\Anaconda3\envs\opanalytics\python.exe";
            string environment = "opanalytics";
            int timeout = 10000;
            PythonRunner pyRunner = new PythonRunner(interpreter, environment, timeout);
            Bitmap bmp = null;

            try
            {
                string command = $"Models/forecastStat.py {attribute}.csv";
                string list = pyRunner.runDosCommands(command);

                if (string.IsNullOrWhiteSpace(list))
                {
                    Console.WriteLine("Error in the script call");
                    goto lend;
                }

                string[] lines = list.Split(new[] { Environment.NewLine }, StringSplitOptions.None);
                string strBitmap = "";
                foreach (string s in lines)
                {
                    if (s.StartsWith("MAPE")) /* Le stringhe che cominciano con "MAPE" sono stringhe di testo che vanno stampate in output (sulla textbox del browser) */
                    {
                        Console.WriteLine(s);
                        res += s;
                    }

                    if (s.StartsWith("b'")) /* Le stringhe che cominciano con "b'" sono stringhe che contengono una codifica binaria di un'immagine */
                    {
                        strBitmap = s.Trim();
                        break;
                    }

                    if (s.StartsWith("Actual")) /* Qui vengono gestite le previsioni, per il momento non viene mai usato */
                    {
                        double fcast = Convert.ToDouble(s.Substring(s.LastIndexOf(" ")));
                        Console.WriteLine(fcast);
                    }
                }

                strBitmap = strBitmap.Substring(strBitmap.IndexOf("b'")); /* Il contenuto di "strBitmap" sarà la codifica binaria delle eventuali immagini stampate da forecastStat.py */
                res += "\",\"img\":\"" + strBitmap + "\""; /* Il contenuto di "res" sarà: "text":"STRINGHE_DI_TESTO_STAMPATE_DA_forecastStat.py","img":"CODIFICA_BINARIA_IMMAGINI_STAMPATE_DA_forecastStat.py" */
                try
                {
                    bmp = pyRunner.FromPythonBase64String(strBitmap);
                }
                catch (Exception exception)
                {
                    throw new Exception(
                        "An error occurred while trying to create an image from Python script output. " +
                        "See inner exception for details.",
                        exception);
                }
                goto lend;
            }
            catch (Exception exception)
            {  
                Console.WriteLine(exception.ToString());
                goto lend;
            }

            lend:
            return res;
        }
    }
}