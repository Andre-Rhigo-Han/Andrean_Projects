"""
Optional SocialBlade scraping utility.

This script is intentionally separated from the modeling pipeline because it
requires a browser session and can be sensitive to website layout changes. Use it
only when follower-count data needs to be refreshed manually.
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager

from data_preparation import find_column, read_table, write_table


class SocialBladeScraper:
    """Search SocialBlade pages and extract the first visible follower/subscriber count."""

    def __init__(self, debugger_address: str = "127.0.0.1:9222", min_sleep: float = 5.0, max_sleep: float = 8.0):
        self.debugger_address = debugger_address
        self.min_sleep = min_sleep
        self.max_sleep = max_sleep
        self.driver = self._connect_driver()

    def _connect_driver(self):
        chrome_options = Options()
        chrome_options.add_experimental_option("debuggerAddress", self.debugger_address)
        return webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

    def _human_pause(self) -> None:
        time.sleep(random.uniform(self.min_sleep, self.max_sleep))

    def search_followers(self, name: str) -> str:
        query = str(name).strip().replace(" ", "+")
        self.driver.get(f"https://socialblade.com/search/search?query={query}")
        self._human_pause()

        xpath_candidates = [
            "//*[contains(text(), 'Followers')]/following-sibling::*[1]",
            "//*[contains(text(), 'Subs')]/following-sibling::*[1]",
            "//*[contains(text(), 'followers')]/following-sibling::*[1]",
            "//*[contains(text(), 'subscribers')]/following-sibling::*[1]",
        ]
        for xpath in xpath_candidates:
            try:
                value = self.driver.find_element(By.XPATH, xpath).text.strip()
                if value:
                    return value
            except Exception:
                continue
        return "Not Found"

    def close(self) -> None:
        # Keep the manually opened browser alive; only detach from WebDriver.
        try:
            self.driver.service.stop()
        except Exception:
            pass


def enrich_social_counts(
    input_path: str | Path,
    output_path: str | Path,
    celebrity_col: str = "celebrity_name",
    partner_col: str = "ballroom_partner",
    debugger_address: str = "127.0.0.1:9222",
) -> pd.DataFrame:
    """Add celebrity and partner follower strings to an input table."""
    df = read_table(input_path)
    celeb_col = find_column(df, [celebrity_col, "Celebrity", "name"])
    partner_col_found = find_column(df, [partner_col, "Partner", "partner_name"], required=False)

    scraper = SocialBladeScraper(debugger_address=debugger_address)
    celebrity_results: list[str] = []
    partner_results: list[str] = []

    try:
        for _, row in df.iterrows():
            celebrity_results.append(scraper.search_followers(str(row[celeb_col])))
            if partner_col_found:
                partner_results.append(scraper.search_followers(str(row[partner_col_found])))
    finally:
        scraper.close()

    df["celebrity_followers_raw"] = celebrity_results
    if partner_col_found:
        df["partner_followers_raw"] = partner_results
    write_table(df, output_path)
    return df


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Refresh social-media follower counts through a manual Chrome session.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="outputs/social_media_results.xlsx")
    parser.add_argument("--celebrity-col", default="celebrity_name")
    parser.add_argument("--partner-col", default="ballroom_partner")
    parser.add_argument("--debugger-address", default="127.0.0.1:9222")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    enrich_social_counts(args.input, args.output, args.celebrity_col, args.partner_col, args.debugger_address)
    print(f"Saved follower search results to {args.output}")


if __name__ == "__main__":
    main()
