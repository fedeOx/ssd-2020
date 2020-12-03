using Microsoft.EntityFrameworkCore;

namespace SsdWebApi
{
    public class IndiciContext : DbContext
    {
        public IndiciContext(DbContextOptions<IndiciContext> options) : base(options)
        {

        }

        public DbSet<Indice> indici { get; set; }

    }
}