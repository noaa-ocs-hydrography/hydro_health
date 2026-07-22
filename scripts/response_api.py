import requests
import pathlib
import pandas as pd

OUTPUTS = pathlib.Path(__file__).resolve().parents[1] / 'outputs'

url = "https://responselink.orr.noaa.gov/hotline/search"

# UPDATE THIS: Paste your fresh cookie string from DevTools
cookie_str = "_ga_9L5LCBL7Q7=GS2.2.s1769027090$o1$g0$t1769027090$j60$l0$h0; nmstat=4565068d-49fb-1c10-abda-171c474adb27; _ga_YF1P2T3NLH=GS2.1.s1769544023$o3$g0$t1769544023$j60$l0$h0; _ga_3F1Q7TDF2C=GS2.1.s1771003795$o2$g1$t1771004006$j29$l0$h0; _4c_=%7B%22_4c_s_%22%3A%22lVNNj9sgEP0rK3oNCRhsQ25VK1U99NaqxxWGsY02MRYmcber%2FPcO%2Bdhks%2B2hOTjDmzeP%2BeKFzD0MZM3rmjMmWVELJhbkCZ4nsn4h0bv8tydrIsDyFoqCOqkllRIkbaR1VBijdA1l3bZAFuTXUUvVta61LEt5WBA7njVeiA0OUIvrJVdLRtsJI9JvRKhgDO0xBrez6TE9j5k3Q%2FMwuSd0ONh7C4%2Bzd6lHR1FW7Ir24Ls%2BZV0pj%2FAY8wGt2Q8uzK9hvKjkFX0N00wj2sQwT5AjP%2FUxbOEB1RAO2Ajy8xiRs43QQoxHWp%2FSOK1Xq3mel10I3QaWNmxXSJp8yukPwRj07M8INvUMtn7qIXrIgs4miNvp0YYhRd%2FsUsjabxx7iKaDEzrmWRRobII1mwzi%2BBb4yfgYHdq7uLnL7vW%2B5SWl1QAzhaHbmMHRrXfUJDSTt6vJehgsUGeSWU27uMdVoEOIqQczJXrP%2FwA2TM9Tgi3dhsFj9n7o6CkOc%2FkefddB%2FAapD7hKeDbOJx8Gs8kl4uZhRx1MvstVuDwNPD2lMN7AXz4%2B%2Fvj6GV1SCF7WQi55XSmhVVlqcrjsHGeaiaJSmuFOJWyBqiTLv8Opa8cVLG7ZBRNK6%2Ffs0yZge%2F4vbO8vb6WphRG6VpSB4lS6sqUGJKMlb1vuTOF425CrJCvrosQHcJbk6qLYNs1FciK3KXBd8qp4n4KD9hKgWmQpsFRjBOUcGmpUIygrZGEsx%2BuYIG%2FLkkzzv2gOF8mbkVyfuWKS8xJ5fkxn4qkogZ6K6TsuIpl7kXT3Wid%2FnK9SJ0clVPWWmpE82nh%2F6z%2Boh8Mf%22%7D; _ga_93J4YEC73J=GS2.1.s1771003793$o2$g1$t1771004027$j8$l0$h0; _ga_HS0NRB74WC=GS2.1.s1772044886$o2$g0$t1772044886$j60$l0$h0; _ga_9WDTQW3Z20=GS2.1.s1773957509$o2$g0$t1773957509$j60$l0$h0; _ga_ZRYFXGJ82Y=GS2.1.s1777042606$o26$g1$t1777044375$j60$l0$h0; _ga_MKKEKN8PK4=GS2.1.s1777042606$o26$g1$t1777044375$j31$l0$h0; _ga_Z9EE7D2SNT=GS2.1.s1777388398$o8$g1$t1777388430$j28$l0$h0; _ga_6L4ZVQ01YN=GS2.1.s1779385588$o1$g1$t1779385611$j37$l0$h0; _ga_E1Q1BML6E5=GS2.1.s1783377757$o1$g1$t1783378023$j60$l0$h0; _ga_SM6YMQ7JH4=GS2.1.s1783377735$o12$g1$t1783378947$j60$l0$h0; _ga_7PDTL9NJYZ=GS2.1.s1783435527$o1$g1$t1783435715$j60$l0$h0; _ga_LJ4WTELRG6=GS2.1.s1783435720$o3$g1$t1783435763$j17$l0$h0; _ga_FHLZQ78JLV=GS2.1.s1783623373$o1$g0$t1783624347$j60$l0$h0; _ga_8QRDKZKW09=GS2.1.s1783697433$o5$g1$t1783697483$j10$l0$h0; _ga_FVJXY24VSZ=GS2.1.s1783957802$o15$g0$t1783957802$j60$l0$h0; _ga=GA1.1.43315734.1768398559; _ga_SSSS43Y0LQ=GS2.1.s1784081846$o8$g1$t1784083790$j60$l0$h0; _ga_44FQGBN8G1=GS2.1.s1784145231$o9$g0$t1784145231$j60$l0$h0; _ga_H4W2NYS81W=GS2.1.s1784662313$o1$g1$t1784662345$j28$l0$h0; _ga_K36L3NSS30=GS2.1.s1784662166$o1$g1$t1784662346$j60$l0$h0; amlbcookie=04; noaasso=FcLBYN_FHUXI2xLi8Mqy4Lvw4JI.*AAJTSQACMDIAAlNLABxJZXlYM25UUEU0cnRkR0NLN3U2TitjNnNmV1E9AAR0eXBlAANDVFMAAlMxAAIwNA..*; session=q8S2XgOVA3WZNpcxbUhA-v-XSgolXaYIE3mZ_xyrVNK3oL0-bkcSRODAtwAsgqYGPOVMyYr_aCpIsj86nlmhfnNlc3Npb246cmxpbms6M1JCc2E2WFhIb3YyNFlsY1Z2WjBSUmtNZFgwN3d2cXZzVkRDN2xGbnVuMVRzaWRzSHozYTZXOEh6VGxtNWI1ZQ; _ga_CSLL4ZEK4L=GS2.1.s1784675386$o76$g1$t1784679351$j60$l0$h0; _ga_070605NEMC=GS2.1.s1784675386$o2$g1$t1784679352$j59$l0$h0"

headers = {
    "accept": "application/json, text/javascript, */*; q=0.01",
    "content-type": "application/json",
    "referer": "https://responselink.orr.noaa.gov/hotline/",
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/150.0.0.0 Safari/537.36 Edg/150.0.0.0",
    "x-requested-with": "XMLHttpRequest",
    "cookie": cookie_str
}

# Payload strictly matching the schema from DevTools
payload = {
    "biorem": None,
    "burn": None,
    "disperse": None,
    "effort": [],
    "geo": False,
    "interactive": True,
    "limit": 1000000,  # High limit to fetch all available records in one request
    "name": [],
    "posts": [],
    "response": [],
    "results": [],
    "shore": None,
    "skim": None,
    "start": "1957-01-01",
    "stop": "2026-12-31",
    "tags": [],
    "text": "",
    "threat": []
}

print("Submitting request...")
response = requests.post(url, headers=headers, json=payload)

print(f"Status Code: {response.status_code}")
print(f"Content-Type: {response.headers.get('Content-Type')}")

if response.status_code == 200 and "application/json" in response.headers.get("Content-Type", ""):
    data = response.json()
    results = data.get('results', [])
    total_count = data.get('count', len(results))

    print("\n--- SUCCESS ---")
    print(f"Server reported total count: {total_count}")
    print(f"Retrieved records: {len(results)}")

    if results:
        OUTPUTS.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(results)
        csv_path = OUTPUTS / 'responselink_incidents_all.csv'
        df.to_csv(csv_path, index=False)
        print(f"\nSaved CSV to:\n{csv_path}")
    else:
        print("No records returned in 'results'.")
else:
    print("\nRequest failed. Check if the session cookie has expired.")
    print("Response preview:")
    print(response.text[:300])