#[derive(Parser, Debug)]
#[command(name = "ceres")]
pub struct Config {
    #[arg(long, env = "DATABASE_URL")]
    pub database_url: String,

    #[arg(long, env = "OPENAI_API_KEY")]
    pub openai_api_key: String,

    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand, Debug)]
pub enum Command {
    /// Harvest data from CKAN portal
    Harvest { portal_url: String },
    /// Searches indexed datasets
    Search { query: String, #[arg(long, default_value = "10")] limit: usize },
}