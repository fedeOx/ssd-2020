using System;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Text;

public class PythonRunner
{
		// strings collecting output/error messages, in case
		public StringBuilder _outputBuilder;
		// private StringBuilder _errorBuilder;

		// The Python interpreter ('python.exe') that is used by this instance.
		public string Interpreter { get; }

		// The timeout for the underlying component in msec.
		public int Timeout { get; set; }

      // The anaconda environment to activate
      public string Environment { get; set; }

      // <param name="interpreter"> Full path to the Python interpreter ('python.exe').
      // <param name="timeout"> The script timeout in msec. Defaults to 10000 (10 sec).
      public PythonRunner(string interpreter, string environment, int timeout = 10000)
		{
			if (interpreter == null)
			{	throw new ArgumentNullException(nameof(interpreter));
			}

			if (!File.Exists(interpreter))
			{	throw new FileNotFoundException(interpreter);
			}

			Interpreter = interpreter;
			Timeout     = timeout;
         Environment = environment;
		}

      // to run a sequence of dos commands, not read from a file
	  /* Funzione che si occupa di fare la chiamata allo script phyton.
	     Fa partire un processo esterno dentro l'environment anaconda.
	     Poiché bisogna passare per anaconda, prima dobbiamo attivare anaconda
	     (l'environment) e poi far partire l'interprete phyton. */
      public string runDosCommands(string strCommand)
      { 
         _outputBuilder = new StringBuilder();
         string res = "";
         var pi = new ProcessStartInfo 
         {
            // Separated FileName and Arguments
            FileName = "cmd.exe", /* Processo da attivare */
            Arguments = $"/c D:/Users/Federico/Anaconda3/condabin/conda.bat activate {Environment}&&python {strCommand}", /* Gli passo il path per arrivare ad anaconda, così può essere attivato l'environmente. "&&" indica un secondo comando che sarà "phyton ...". */
            UseShellExecute = false, 
            CreateNoWindow = false,
            ErrorDialog = false,
            RedirectStandardError = true,
            RedirectStandardOutput = true,
            RedirectStandardInput = true,
         };         

         using (var process = new Process(){
            StartInfo = pi, /* Attivazione di un nuovo processo. Il processo attivato è quello che viene indicato dentro le ProcessStartInfo contenute nella variabile "pi". */
            EnableRaisingEvents = true
            })
         {
            process.OutputDataReceived += (sender, e) => /* Qui si definisce un handler per gli eventi di output prodotti dal processo. */
            {
               // could be null terminated, needs null handling
               if (e.Data != null)
               {
                  //Console.WriteLine("> "+e.Data);
                  _outputBuilder.AppendLine(e.Data); /* Ogni volta che viene prodotta una stringa dal processo, essa viene aggiunta in output a questo _outputBuilder. */
               }
            };
         
            process.Exited += (sender, e) => /* Qui si definisce un handler per l'evento di chiusura del processo. */
            {
               // when Exited is called, OutputDataReceived could still being loaded
               // you need a proper release code here
               Console.WriteLine("exiting ...");
               res = _outputBuilder.ToString(); /* Quando il processo termina, concatento tutto quello che era stato appeso dentro _outputBuilder per creare un'unica stringona. */
            };
         
            process.Start(); /* Qui viene fatto partire il processo. */
            // You need to call this explicitly after Start
            process.BeginOutputReadLine(); /* Dice di leggere tutto quello che viene prodotto in output dal processo */

            /*
            // Pass multiple commands to cmd.exe
            using (var sw = process.StandardInput)
            {
               if (sw.BaseStream.CanWrite)
               {
                  //sw.WriteLine("echo off");
                  sw.WriteLine($"/ProgramData/Anaconda3/condabin/conda.bat activate {Environment}");
                  sw.WriteLine($"python {strCommand}");
                  //sw.WriteLine("exit");
               }
            }  
            */    

            // With WaitForExit, it is same as synchronous,
            // to make it truly asynchronous, you'll need to work on it from here
            process.WaitForExit();
         }
         // here no more process
         return res;
      }    

      // Converts a base64 string (as printed by python script) to a bitmap image.
      public Bitmap FromPythonBase64String(string pythonBase64String)
      {
         // Remove the first two chars and the last one.
         // First one is 'b' (python format sign), others are quote signs.
         string base64String = pythonBase64String.Substring(2, pythonBase64String.Length - 3);

         // Convert now raw base46 string to byte array.
         byte[] imageBytes = Convert.FromBase64String(base64String);

         // Read bytes as stream.
         var memoryStream = new MemoryStream(imageBytes, 0, imageBytes.Length);
         memoryStream.Write(imageBytes, 0, imageBytes.Length);

         // Create bitmap from stream.
         return (Bitmap)Image.FromStream(memoryStream, true);
      }
}