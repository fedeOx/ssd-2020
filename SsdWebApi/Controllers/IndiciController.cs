using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using Microsoft.Data.Sqlite;

namespace SsdWebApi.Controllers
{
    [ApiController]
    [Route("api/Indici")] // [Route("api/[controller]")]
    public class IndiciController : ControllerBase
    {
        private readonly IndiciContext _context;
        private Persistance persistence;

        public IndiciController(IndiciContext context)
        {
            _context = context;
            persistence = new Persistance(context);
        }
    
        [HttpGet]
        public ActionResult<List<Indice>> GetAll() => _context.indici.ToList();

        [HttpGet("{id}")] //[HttpGet("{id}", Name = "GetSerie")]
        public string GetSerie(int id)
        {
            if (id > 6) id = 6;

            string res = "{";
            string[] indices = new string[] {"SP_500", "FTSE_MIB", "GOLD_SPOT", "MSCI_EM", "MSCI_EURO", "All_Bonds", "US_Treasury"};
            string attribute = indices[id]; // In "attribute" ci sarà l'indice richiesto (esso sarà anche il nome del file ".csv" che andrà a creare)

            persistence.readIndexAndBuildCSV(attribute);

            Forecast forecast = new Forecast();
            res += forecast.forecastSARIMAindex(attribute);
            res += "}";
           
            return res;
        }

    }
}