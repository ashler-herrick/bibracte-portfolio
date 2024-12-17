from bet_edge.odds_api.api_client import APIClient
from bet_edge.data_io.env_credential_provider import EnvironmentCredentialProvider

credential_provider = EnvironmentCredentialProvider()
api_client = APIClient(credential_provider, sport_code="americanfootball_nfl")

res = api_client.fetch_sports()
print(res)
print("Result of fetch_sports()^^")

res = api_client.fetch_current_event_ids()
print(res)
print("Result of fetch_current_event_ids()^^")
event_id = res[0]["id"]

res = api_client.fetch_data_for_current_event(event_id, "h2h")
print(res)
print("Result of fetch_data_for_current_event()^^")

res = api_client.fetch_historical_event_ids("2023-12-06")
print(res)
print("Result of fetch_historical_event_ids()^^")


hist_event_id = res["data"][0]["id"]
print(hist_event_id)

res = api_client.fetch_data_for_historical_event(hist_event_id, "2023-12-06", "h2h")
print(res)
print("Result of fetch_data_for_historical_event()^^")
