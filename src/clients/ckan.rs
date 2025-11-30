pub struct CkanClient {
    client: reqwest::Client,
    base_url: Url,
}

impl CkanClient {
    pub async fn list_packages(&self) -> Result<Vec<String>, AppError>;
    pub async fn get_package(&self, id: &str) -> Result<CkanPackage, AppError>;
    pub fn stream_packages(&self) -> impl Stream<Item = Result<CkanPackage, AppError>>;
}