from tenacity import retry, wait_exponential_jitter, stop_after_attempt

# Decorador default para llamadas externas
retry_external = retry(
    wait=wait_exponential_jitter(initial=1, max=30),
    stop=stop_after_attempt(5)
)
