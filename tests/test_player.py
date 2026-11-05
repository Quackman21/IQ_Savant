from datetime import date

from core.player import age_ym, bio, draft_entry, money, ordinal, short_date, signing_entry


def test_formats():
    assert [ordinal(n) for n in (1, 2, 3, 11, 19, 22, 113)] == ["1st", "2nd", "3rd", "11th", "19th", "22nd", "113th"]
    assert (money("8100000.00"), money(3400000), money(88000), money(1000000), money(None)) == \
        ("$8.1M", "$3.4M", "$88K", "$1M", "")
    assert age_ym("2003-07-05", date(2026, 11, 5)) == "23.04"
    assert age_ym("2002-03-25", date(2026, 11, 5)) == "24.07"
    assert short_date("1998-02-06") == "2/6/98"


def test_draft_entries_match_examples():
    rutschman = {"pickRound": "1", "pickNumber": 1, "signingBonus": "8100000.00",
                 "school": {"name": "Oregon State", "schoolClass": "4YR JR", "state": "OR"}}
    assert draft_entry(rutschman, 2019, "BAL") == \
        "2019 by BAL, Round 1, 1st Overall, Oregon State, Signing Bonus $8.1M"
    pca = {"pickRound": "1", "pickNumber": 19, "signingBonus": "3359000.00",
           "school": {"name": "Harvard-Westlake", "schoolClass": "HS SR", "state": "CA"}}
    assert draft_entry(pca, 2020, "NYM") == \
        "2020 by NYM, Round 1, 19th Overall, Harvard-Westlake HS (CA), Signing Bonus $3.4M"


def test_international_signing():
    tx = [{"typeCode": "TR", "date": "2022-11-17", "toTeam": {"id": 139}},
          {"typeCode": "SFA", "date": "2019-07-02", "toTeam": {"id": 114, "name": "Cleveland Guardians"}}]
    abbr = {114: "CLE", 139: "TB"}.get
    assert signing_entry(tx, "Dominican Republic", abbr) == "Signed by CLE out of Dominican Republic"
    assert signing_entry([], "Venezuela", abbr) is None


def test_bio_fields():
    p = {"firstName": "Junior", "lastName": "Caminero", "primaryPosition": {"abbreviation": "3B"},
         "height": "6' 1\"", "weight": 220, "active": True, "batSide": {"code": "R"},
         "pitchHand": {"code": "R"}, "birthDate": "2003-07-05", "birthCity": "Bani",
         "birthCountry": "Dominican Republic", "currentTeam": {"id": 139, "name": "Tampa Bay Rays"}}
    b = bio(p, date(2026, 11, 5))
    assert (b["first"], b["last"], b["pos"], b["hw"], b["status"], b["bt"], b["age"], b["dob"]) == \
        ("Junior", "Caminero", "3B", "6'1/220", "ACTIVE", "R/R", "23.04", "7/5/03")
    assert b["born"] == "Bani, Dominican Republic" and b["debut"] == "-"


def test_summary_tabs_by_position():
    from core.player import position_games, summary_tabs
    splits = [{"season": "2026", "position": {"abbreviation": "C"}, "stat": {"gamesPlayed": 90}},
              {"season": "2026", "position": {"abbreviation": "1B"}, "stat": {"gamesPlayed": 12}},
              {"season": "2020", "position": {"abbreviation": "LF"}, "stat": {"gamesPlayed": 40}}]
    g = position_games(splits, since=2024)
    assert g == {"C": 90, "1B": 12}                      # 2020 LF too old
    assert summary_tabs("C", g, 2000) == ["Hitting", "Core 4", "Defense", "Baserunning"]
    assert summary_tabs("CF", {"CF": 150}, 600) == ["Hitting", "Core 4", "Defense", "Baserunning"]
    from core.defense import defense_tabs
    assert defense_tabs("C", g) == ["Catching", "Infield"]
    assert defense_tabs("SS", {"SS": 100, "2B": 30, "LF": 15}) == ["Infield", "Outfield"]
    assert defense_tabs("CF", {"CF": 150}) == ["Outfield"]
    assert summary_tabs("P", {"P": 60}, 0) == ["Pitching"]
    assert summary_tabs("TWP", {"P": 20}, 1500)[0] == "Pitching"
    assert "Hitting" in summary_tabs("TWP", {"P": 20}, 1500)
    assert summary_tabs("DH", {}, 600) == ["Hitting", "Core 4", "Baserunning"]
