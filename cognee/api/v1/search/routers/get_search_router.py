from typing import Optional
from uuid import UUID
from datetime import datetime
from typing import Union, Optional
from fastapi import Depends, APIRouter
from fastapi.responses import JSONResponse
from cognee.modules.search.types import SearchType
from cognee.api.DTO import InDTO, OutDTO
from cognee.modules.users.models import User
from cognee.modules.search.operations import get_history
from cognee.modules.users.methods import get_authenticated_user
from cognee.modules.engine.models import NodeSet
import traceback




class SearchPayloadDTO(InDTO):
    search_type: SearchType
    query: str
    datasets: Union[list[str], str, None] = None
    node_name: Optional[list[str]] = None
    node_type: Optional[str] = None
    top_k: Optional[int] = 10


def get_search_router() -> APIRouter:
    router = APIRouter()

    class SearchHistoryItem(OutDTO):
        id: UUID
        text: str
        user: str
        created_at: datetime

    @router.get("/", response_model=list[SearchHistoryItem])
    async def get_search_history(user: User = Depends(get_authenticated_user)):
        try:
            history = await get_history(user.id)

            return history
        except Exception as error:
            return JSONResponse(status_code=500, content={"error": str(error)})

    @router.post("/", response_model=list)
    async def search(payload: SearchPayloadDTO, user: User = Depends(get_authenticated_user)):
        """This endpoint is responsible for searching for nodes in the graph."""
        from cognee.api.v1.search import search as cognee_search

        try:
            results = await cognee_search(
                query_text=payload.query, 
                query_type=payload.search_type, 
                user=user,
                datasets=payload.datasets,
                node_name=payload.node_name,
                node_type=NodeSet,
                top_k=payload.top_k,
            )

            return results
        except Exception as error:
            print(traceback.format_exc())
            return JSONResponse(status_code=409, content={"error": str(error)})

    return router
