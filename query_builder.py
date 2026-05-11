# query_builder.py


# ---------------- SELECT 생성 ----------------

def build_select_clause(aggregation):

    # 집계 없음
    if not aggregation:

        return "SELECT *"


    agg_type = aggregation.get(
        "type"
    )

    # ---------------- Normalization ----------------

    if isinstance(agg_type, str):

        agg_type = agg_type.upper()


    column = aggregation.get(
        "column"
    )


    # ---------------- COUNT ----------------

    if agg_type == "COUNT":

        return f"SELECT COUNT({column}) AS result"


    # ---------------- AVG ----------------

    elif agg_type == "AVG":

        return f"SELECT AVG({column}) AS result"


    # ---------------- SUM ----------------

    elif agg_type == "SUM":

        return f"SELECT SUM({column}) AS result"


    # ---------------- MAX ----------------

    elif agg_type == "MAX":

        return f"SELECT MAX({column}) AS result"


    # ---------------- MIN ----------------

    elif agg_type == "MIN":

        return f"SELECT MIN({column}) AS result"


    return "SELECT *"


# ---------------- WHERE 생성 ----------------

def build_where_clause(filters):

    if not filters:

        return ""


    conditions = []


    for f in filters:

        column = f["column"]

        op = f["op"]

        value = f["value"]


        # ---------------- 문자열 처리 ----------------

        if isinstance(value, str):

            value = f"'{value}'"


        conditions.append(

            f"{column} {op} {value}"
        )


    return "WHERE " + " AND ".join(conditions)


# ---------------- ORDER BY 생성 ----------------

def build_order_by_clause(sort):

    if not sort:

        return ""


    column = sort.get(
        "column"
    )

    direction = sort.get(
        "direction"
    )


    # ---------------- Normalization ----------------

    if isinstance(direction, str):

        direction = direction.upper()


    return f"ORDER BY {column} {direction}"


# ---------------- LIMIT 생성 ----------------

def build_limit_clause(limit):

    if not limit:

        return ""


    return f"LIMIT {limit}"


# ---------------- SQL 생성 ----------------

def build_sql(ast):

    table = ast["table"]

    aggregation = ast.get(
        "aggregation"
    )

    filters = ast.get(
        "filters",
        []
    )

    sort = ast.get(
        "sort"
    )

    limit = ast.get(
        "limit"
    )


    # ---------------- SELECT ----------------

    select_clause = build_select_clause(
        aggregation
    )


    # ---------------- WHERE ----------------

    where_clause = build_where_clause(
        filters
    )


    # ---------------- ORDER BY ----------------

    order_by_clause = build_order_by_clause(
        sort
    )


    # ---------------- LIMIT ----------------

    limit_clause = build_limit_clause(
        limit
    )


    # ---------------- SQL 조립 ----------------

    sql = f"""
    {select_clause}
    FROM {table}
    {where_clause}
    {order_by_clause}
    {limit_clause}
    """


    return sql.strip()