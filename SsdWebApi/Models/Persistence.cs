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

        public void testDB()
        {
            var connectionStringBuilder = new SqliteConnectionStringBuilder();
            //Use DB in project directory. If it does not exist, create it:
            connectionStringBuilder.DataSource = "./testDB.sqlite";
            using (var connection = new SqliteConnection(connectionStringBuilder.ConnectionString))
            {
                connection.Open();
                //Create a table (drop if already exists):
                var delTableCmd = connection.CreateCommand();
                delTableCmd.CommandText = "DROP TABLE IF EXISTS cronistoria";
                delTableCmd.ExecuteNonQuery();
                var createTableCmd = connection.CreateCommand();
                createTableCmd.CommandText = "CREATE TABLE cronistoria(id INTEGER PRIMARY KEY, anno int, serie text)";
                createTableCmd.ExecuteNonQuery();
                //Seed some data:
                using (var transaction = connection.BeginTransaction())
                {
                    var insertCmd = connection.CreateCommand();
                    insertCmd.CommandText = "INSERT INTO cronistoria (anno,serie) VALUES(2014,'A')";
                    insertCmd.ExecuteNonQuery();
                    insertCmd.CommandText = "INSERT INTO cronistoria (anno,serie) VALUES(2015,'B')";
                    insertCmd.ExecuteNonQuery();
                    insertCmd.CommandText = "INSERT INTO cronistoria (anno,serie) VALUES(2016,'B')";
                    insertCmd.ExecuteNonQuery();
                    insertCmd.CommandText = "INSERT INTO cronistoria (anno,serie) VALUES(2017,'B')";
                    insertCmd.ExecuteNonQuery();
                    insertCmd.CommandText = "INSERT INTO cronistoria (anno,serie) VALUES(2018,'D')";
                    insertCmd.ExecuteNonQuery();
                    transaction.Commit();
                }
                //Read the newly inserted data:
                var selectCmd = connection.CreateCommand();
                selectCmd.CommandText = "SELECT anno, serie FROM cronistoria";
                using (var reader = selectCmd.ExecuteReader())
                {
                    while (reader.Read())
                    {
                        int anno = Convert.ToInt32(reader.GetString(0));
                        var message = anno + " " + (anno + 1) + " Serie " + reader.GetString(1);
                        Console.WriteLine(message);
                    }
                }
            }
        }
    }
}