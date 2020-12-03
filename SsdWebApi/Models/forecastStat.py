import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import os, sys, io, base64

def print_figure(fig):
    # Leggi prima i commenti sotto a questa funzione.
    # In pratica converte l'immagine passata in input in una stringa di caratteri (perché deve viaggiare in Internet
    # e in Internet viaggiano solo stringhe di caratteri, non posso mandare l'immagine come stringa binaria ma devo
    # convertirne il formato in base64).
	"""
	Converts a figure (as created e.g. with matplotlib or seaborn) to a png image and this 
	png subsequently to a base64-string, then prints the resulting string to the console.
	"""
	buf = io.BytesIO()
	fig.savefig(buf, format='png')
	print(base64.b64encode(buf.getbuffer()))
    # Notare che la stringa prodotta, comincia con "b'" e termina con "'". Tali caratteri non fanno parte della codifica
    # base64 dell'immagine quindi dovremo ad un certo punto toglierli entrambi.


if __name__ == "__main__":
   # change working directory to script path
   abspath = os.path.abspath(__file__)
   dname = os.path.dirname(abspath)
   os.chdir(dname)

   # Qui vengono stampati i parametri che vengono passatti all'atto della messa in esecuzione.
   # Si vada su Run > Configuration per file > General Settings > spunta su Command line options > specificare il file csv su cui lavorare (es. FTSE_MIB.csv)
   # Impostare tali file csv in questo modo serve solo per test, in realtà noi faremo in modo che sia il C# a dirgli
   # quale file csv andare a leggere.
   print('MAPE Number of arguments:', len(sys.argv)) # Scrive la lunghezza del vettore degli argomenti (argv).
   print('MAPE Argument List:', str(sys.argv), ' first true arg:',sys.argv[1]) # Scrive la lista degli argomenti seguito dal secondo argomento di argv che sarebbe il file csv (il primo è il file pythos stesso).
   
   """
   L'idea è che nel progetto vero, così come vengono scritte le stringhe qui sopra, il nostro script phyton produrrà
   delle stringhe che rappresentano le varie previsioni. Queste previsioni dovranno essere lette dal server.
   Potrebbero anche bastare le stringhe. Visto che ci siamo qui sotto viene mostrato anche come generare e passare
   delle immagini.
   
   Il nostro script phyton quindi riceverà in input il nome del file degli indici della serie storica su cui dovrà lavorare,
   poi dovrà elaborare quel file e fare su di esso delle previsioni. Le previsioni dovranno essere stampate (potrà essere
   necessario stampare stringhe e/o immagini). L'idea è che metteremo in esecuzione tale script dal server C#, quindi
   l'output prodotto dovrà essere catturato dal server e utilizzato per mandarlo al client che lo visualizzerà.
   """
   
   dffile = sys.argv[1] # recupero il file che voglio andare a leggere
   df = pd.read_csv("../"+dffile) # leggo il contenuto del file
   
   plt.plot(df) # faccio il grafico dei dati contenuti nel file. Qui viene effettivamente generata un'immagine.
   #plt.show()
   
   # Finally, print the chart as base64 string to the console.
   print_figure(plt.gcf()) # Qui viene chiamata la funzione print_figure passando in input plt.gcf(). gcf() va a prendere la figura corrente (ovvero quella creata nella riga sopra). Vedi commenti nella funzione.
   

   