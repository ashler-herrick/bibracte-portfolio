from bet_edge.data_io.env_credential_provider import EnvironmentCredentialProvider


def test_env_cred_manager():
    env_cred_manager = EnvironmentCredentialProvider()
    creds = env_cred_manager.get_credentials()
    assert creds != {}
