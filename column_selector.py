# column_selector.py

from config.validation_config import (
    VALID_COLUMNS,
    COLUMN_SYNONYMS,
    AGGREGATION_KEYWORDS
)


class ColumnSelector:

    def __init__(self):

        self.columns = VALID_COLUMNS

    def detect_aggregation(
        self,
        query
    ):

        for keyword, agg in AGGREGATION_KEYWORDS.items():

            if keyword in query:

                return agg

        return None


    def select_columns(
        self,
        query
    ):

        selected = set()


        # ---------------- Synonym Matching ----------------

        for column, synonyms in COLUMN_SYNONYMS.items():

            for synonym in synonyms:

                if synonym in query:

                    selected.add(column)


        # ---------------- Direct Match ----------------

        for column in self.columns:

            if column in query:

                selected.add(column)


        aggregation = self.detect_aggregation(
            query
        )


        return {

            "columns": list(selected),

            "aggregation": aggregation
        }