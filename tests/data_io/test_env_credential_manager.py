from bet_edge.data_io.env_cred_provider import EnvCredProvider


def test_env_cred_manager():
    env_cred_manager = EnvCredProvider()
    creds = env_cred_manager.get_credentials()
    assert creds != {}
