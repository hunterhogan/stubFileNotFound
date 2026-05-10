import bigframes.pandas as bpd
from bigframes import dtypes as dtypes
from google.auth import credentials as credentials
from typing import TypedDict

class TableReference(TypedDict):
    project_id: str
    dataset_id: str
    table_id: str

class TableSchemaEntry(TypedDict):
    name: str
    field_type: str
    mode: str | None
    description: str | None

class ValidationError(TypedDict):
    message: str
    line: int | None
    column: int | None

class ValidationSuccess(TypedDict):
    bytes_processed: int
    compiled_sql: str
    tables: list[TableReference]
    schema: list[TableSchemaEntry]

class ValidationFailure(TypedDict):
    authorization_failed: bool
    errors: list[ValidationError]
ValidationResult = ValidationSuccess | ValidationFailure

def set_credentials(creds: credentials.Credentials | None = None, project_id: str | None = None): ...
def validate(sql: str) -> ValidationResult: ...
def run(sql: str) -> bpd.DataFrame: ...
