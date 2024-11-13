# Databricks notebook source
def get_settings(target: str) -> dict[str, str]:
    match target:
        case "TEST":
            settings = {"table_suffix": "_test",
                        "limit": "{$limit: 20},"}

        case "PROD":
            settings = {"table_suffix": "",
                        "limit": ""}

        case _:
            raise ValueError("TARGET must be either TEST or PROD")

    print("Succesfully retrieved settings.")
    return settings
