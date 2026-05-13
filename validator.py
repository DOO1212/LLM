# validator.py

from config.validation_config import (

    VALID_COLUMNS,
    VALID_FILTER_OPERATORS,
    VALID_SORT_DIRECTIONS,
    VALID_AGGREGATIONS
)


# ---------------- Constants ----------------

VALID_TABLES = [

    "inventory"
]

MAX_LIMIT = 100


# ---------------- AST Validation ----------------

def validate_ast(ast):


    # ---------------- AST Type ----------------

    if not isinstance(ast, dict):

        return {

            "valid": False,

            "reason": "INVALID_AST_FORMAT"
        }


    # ---------------- table ----------------

    table = ast.get(
        "table"
    )

    if table not in VALID_TABLES:

        return {

            "valid": False,

            "reason": "INVALID_TABLE"
        }


    # ---------------- aggregation ----------------

    aggregation = ast.get(
        "aggregation"
    )

    if aggregation:

        # aggregation 구조 검증
        if not isinstance(aggregation, dict):

            return {

                "valid": False,

                "reason": "INVALID_AGGREGATION_FORMAT"
            }


        agg_type = aggregation.get(
            "type"
        )

        agg_column = aggregation.get(
            "column"
        )


        # aggregation type 검증
        if agg_type not in VALID_AGGREGATIONS:

            return {

                "valid": False,

                "reason": "INVALID_AGGREGATION_TYPE"
            }


        # aggregation column 검증
        if agg_column not in VALID_COLUMNS:

            return {

                "valid": False,

                "reason": "INVALID_AGGREGATION_COLUMN"
            }


    # ---------------- filters ----------------

    filters = ast.get(
        "filters",
        []
    )

    if not isinstance(filters, list):

        return {

            "valid": False,

            "reason": "INVALID_FILTERS_FORMAT"
        }


    for f in filters:

        # filter 구조 검증
        if not isinstance(f, dict):

            return {

                "valid": False,

                "reason": "INVALID_FILTER_FORMAT"
            }


        column = f.get(
            "column"
        )

        op = f.get(
            "op"
        )

        value = f.get(
            "value"
        )


        # 컬럼 검증
        if column not in VALID_COLUMNS:

            return {

                "valid": False,

                "reason": "INVALID_FILTER_COLUMN"
            }


        # 연산자 검증
        if op not in VALID_FILTER_OPERATORS:

            return {

                "valid": False,

                "reason": "INVALID_FILTER_OPERATOR"
            }


        # value 존재 여부
        if value is None:

            return {

                "valid": False,

                "reason": "INVALID_FILTER_VALUE"
            }


    # ---------------- sort ----------------

    sort = ast.get(
        "sort"
    )

    if sort:

        # sort 구조 검증
        if not isinstance(sort, dict):

            return {

                "valid": False,

                "reason": "INVALID_SORT_FORMAT"
            }


        sort_column = sort.get(
            "column"
        )

        direction = sort.get(
            "direction"
        )


        # 컬럼 검증
        if sort_column not in VALID_COLUMNS:

            return {

                "valid": False,

                "reason": "INVALID_SORT_COLUMN"
            }


        # 방향 검증
        if direction not in VALID_SORT_DIRECTIONS:

            return {

                "valid": False,

                "reason": "INVALID_SORT_DIRECTION"
            }


    # ---------------- limit ----------------

    limit = ast.get(
        "limit"
    )

    if limit is not None:

        if not isinstance(limit, int):

            return {

                "valid": False,

                "reason": "INVALID_LIMIT"
            }


        if limit <= 0:

            return {

                "valid": False,

                "reason": "INVALID_LIMIT"
            }


        if limit > MAX_LIMIT:

            return {

                "valid": False,

                "reason": "LIMIT_EXCEEDED"
            }


    # ---------------- selected_columns ----------------

    selected_columns = ast.get(
        "selected_columns",
        []
    )

    if not isinstance(selected_columns, list):

        return {

            "valid": False,

            "reason": "INVALID_SELECTED_COLUMNS_FORMAT"
        }


    for column in selected_columns:

        if column not in VALID_COLUMNS:

            return {

                "valid": False,

                "reason": "INVALID_SELECTED_COLUMN"
            }


    # ---------------- return_column ----------------

    return_column = ast.get(
        "return_column"
    )

    if return_column:

        if return_column not in VALID_COLUMNS:

            return {

                "valid": False,

                "reason": "INVALID_RETURN_COLUMN"
            }


    # ---------------- 통과 ----------------

    return {

        "valid": True,

        "reason": None
    }