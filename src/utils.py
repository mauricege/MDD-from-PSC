import pandas as pd
import regex as re
from collections import defaultdict
from typing import Any, Iterable

GROUP_REGEX = r"(?P<subjectID>\w+)/(?P<t>\w+).(.*).wav"


def update_dict_with_list_values(
    a: dict[Any, list],
    b: dict[Any, Any],
) -> dict[Any, list]:
    """Update a dictionary and its list values from another dict."""

    union = defaultdict(list, a)

    for k, value in b.items():
        union[k].append(value)

    return union


def add_groups_to_dataframe(df, group_regex):
    filenames = list(df.values[:, 0])
    group_dict = {}
    for fn in filenames:
        group_dict = update_dict_with_list_values(
            group_dict, re.match(group_regex, fn).groupdict()
        )
    group_dict["interviewType"] = [
        interview_type if interview_type is not None else "hamilton"
        for interview_type in group_dict["interviewType"]
    ]
    group_df = pd.DataFrame(data=group_dict)
    return pd.concat((df, group_df), axis=1)
