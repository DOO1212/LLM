# validator.py

from config.validation_config import (

    VALID_COLUMNS,
    VALID_FILTER_OPERATORS
)


# ---------------- AST Validation ----------------

def validate_ast(ast):

    filters = ast.get(
        "filters",
        []
    )

    sort = ast.get(
        "sort"
    )

    presentation_order = ast.get(
        "presentation_order"
    )

    aggregation = ast.get(
        "aggregation"
    )

    limit = ast.get(
        "limit"
    )


    # ---------------- aggregation ----------------

    if aggregation:

        agg_column = aggregation.get(
            "column"
        )

        if agg_column not in VALID_COLUMNS:

            return {

                "valid": False,

                "reason": "INVALID_AGGREGATION_COLUMN"
            }


    # ---------------- filters ----------------

    for f in filters:

        column = f.get("column")

        op = f.get("op")


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


    # ---------------- sort ----------------

    if sort:

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
        if direction not in ["ASC", "DESC"]:

            return {

                "valid": False,

                "reason": "INVALID_SORT_DIRECTION"
            }


    # ---------------- presentation_order ----------------

    if presentation_order:

        # 현재는 multi-stage sorting 미지원

        return {

            "valid": False,

            "reason": "MULTI_STAGE_SORT_NOT_SUPPORTED"
        }


    # ---------------- limit ----------------

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


    # ---------------- 통과 ----------------

    return {

        "valid": True,

        "reason": None
    }