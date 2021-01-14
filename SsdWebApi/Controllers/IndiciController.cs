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
        public string GetPortfolio([FromQuery] int capital, [FromQuery] double riskAlpha)
        {
            Console.WriteLine("Request received");

            if (riskAlpha > 1.0)
                riskAlpha = 1.0;
            if (riskAlpha < 0)
                riskAlpha = 0;

            string res = "{";
            string[] indices = new string[] {"All_Bonds", "FTSE_MIB", "GOLD_SPOT", "MSCI_EM", "MSCI_EURO", "SP_500", "US_Treasury"};
            string sourcesNames = "";
            foreach (string index in indices) {
                persistence.readIndexAndBuildCSV(index);
                sourcesNames += index + ".csv ";
            }

            Forecast forecast = new Forecast();
            res += forecast.predictPortfolio(sourcesNames, capital, riskAlpha);
            res += "}";

            Console.WriteLine("Sending response...");
            return res;
        }

    }
}