using System.IO;
using System.Collections.Generic;
using System;
using Microsoft.Data.Sqlite;
using Microsoft.EntityFrameworkCore;

namespace SsdWebApi
{
    public class Persistance
    {
        private readonly IndiciContext _context;

        public Persistance(IndiciContext context)
        {
            _context = context;
        }

        // legge uno specifico indice da DB
        public List<string>  readIndexAndBuildCSV(string attribute)
        {
            List<string> serie = new List<string>();
            StreamWriter sw = new StreamWriter(attribute + ".csv", false);

            serie.Add(attribute);
            sw.WriteLine(attribute);
            using (var command = _context.Database.GetDbConnection().CreateCommand())
            {
                command.CommandText = $"SELECT {attribute} FROM indici";
                _context.Database.OpenConnection();
                using (var reader = command.ExecuteReader())
                {
                    while (reader.Read())
                    {
                        sw.WriteLine(reader[attribute]);
                        serie.Add(reader[attribute].ToString());
                    }
                }
            }
            sw.Close();

            return serie;
        }
    }
}