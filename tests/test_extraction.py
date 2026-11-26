from maintidoc.extraction import table_to_markdown


def test_table_to_markdown() -> None:
    table = [["Programme", "Dépenses"], ["Education", "100"], ["Health", None]]
    result = table_to_markdown(table)
    assert "| Programme | Dépenses |" in result
    assert "| Education | 100 |" in result
    assert "| Health |  |" in result


def test_empty_table() -> None:
    assert table_to_markdown([]) == ""
    assert table_to_markdown([[None, None]]) == ""
    assert table_to_markdown([["En-tête 1", "En-tête 2"]]) == ""
