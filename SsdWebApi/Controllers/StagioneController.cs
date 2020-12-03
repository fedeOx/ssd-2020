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
    [Route("api/Stagione")]
    public class StagioneController : ControllerBase
    {
        private readonly StagioneContext _context;

        public StagioneController(StagioneContext context)
        {
            _context = context;
        }

        [HttpGet]
        public ActionResult<List<Stagione>> GetAll() => _context.cronistoria.ToList();

        // GET by ID action
        [HttpGet("{id}")]
        public async Task<ActionResult<Stagione>> GetStagione(int id)
        {
            var stagioneItem = await _context.cronistoria.FindAsync(id);
            if (stagioneItem == null)
            {
                return NotFound();
            }
            return stagioneItem;
        }

        // POST action
        [HttpPost]
        [Route("[action]")]
        public string PostStagioneItem([FromBody] Stagione item)
        {
            string res = "Anno " + item.anno;
            try
            {
                _context.cronistoria.Add(item);
                _context.SaveChangesAsync();
            }
            catch (Exception ex)
            {
                Console.WriteLine("[ERROR] " + ex.Message);
                res = "Error";
            }
            Console.WriteLine(res);
            return res;
        }

        // PUT action
        [HttpPut("{id}")]
        public async Task<IActionResult> PutStagione(int id, [FromBody] Stagione item)
        {
            if (id != item.id) return BadRequest();
            _context.Entry(item).State = EntityState.Modified;
            try
            {
                await _context.SaveChangesAsync();
            }
            catch (Exception ex)
            {
                if (!_context.cronistoria.Any(s => s.id == id))
                {
                    return NotFound();
                }
                else
                {
                    Console.WriteLine("[ERROR] " + ex.Message); ;
                }
            }
            return Ok();
        }

        // DELETE action
        [HttpDelete("{id}")]
        public async Task<ActionResult<Stagione>> DeleteItem(int id)
        {
            var item = await _context.cronistoria.FindAsync(id);
            if (item == null)
            {
                return NotFound();
            }
            _context.cronistoria.Remove(item);
            await _context.SaveChangesAsync();
            return item;
        }
    }
}