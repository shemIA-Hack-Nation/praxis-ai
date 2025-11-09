"""
WebSocket endpoints for real-time paper scraping visualization.
"""
import logging
import json
from typing import Dict, Any
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)

router = APIRouter()

# Store active connections
class ConnectionManager:
    """Manages WebSocket connections for real-time updates."""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str):
        """Accept a new WebSocket connection."""
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.info(f"WebSocket connected: {client_id}")
    
    def disconnect(self, client_id: str):
        """Remove a WebSocket connection."""
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            logger.info(f"WebSocket disconnected: {client_id}")
    
    async def send_personal_message(self, message: Dict[str, Any], client_id: str):
        """Send message to a specific client."""
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_json(message)
            except Exception as e:
                logger.error(f"Error sending message to {client_id}: {str(e)}")
                self.disconnect(client_id)
    
    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast message to all connected clients."""
        disconnected = []
        for client_id, connection in self.active_connections.items():
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting to {client_id}: {str(e)}")
                disconnected.append(client_id)
        
        for client_id in disconnected:
            self.disconnect(client_id)

# Global connection manager
manager = ConnectionManager()


@router.websocket("/ws/scraping/{client_id}")
async def websocket_scraping(websocket: WebSocket, client_id: str):
    """
    WebSocket endpoint for real-time paper scraping visualization.
    
    Events:
    - extraction_started: PDF content extraction started
    - pdf_parsed: PDF parsed successfully
    - content_extracted: Content extracted from paper
    - search_term_extraction_started: LLM search term extraction started
    - search_term_extraction_complete: Search terms extracted
    - search_started: arXiv search started
    - paper_found: A paper was found (sent for each paper)
    - search_complete: Search completed
    - ranking_complete: Papers ranked and filtered
    """
    await manager.connect(websocket, client_id)
    
    try:
        while True:
            # Keep connection alive and listen for client messages
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
                logger.info(f"Received message from {client_id}: {message}")
                
                # Echo back or handle client messages
                if message.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON from {client_id}: {data}")
    
    except WebSocketDisconnect:
        manager.disconnect(client_id)
        logger.info(f"Client {client_id} disconnected")
    except Exception as e:
        logger.error(f"WebSocket error for {client_id}: {str(e)}")
        manager.disconnect(client_id)


async def notify_scraping_progress(client_id: str, event: str, data: Dict[str, Any]):
    """Notify a specific client about scraping progress."""
    import datetime
    message = {
        "type": "scraping_progress",
        "event": event,
        "data": data,
        "timestamp": datetime.datetime.now().isoformat()
    }
    await manager.send_personal_message(message, client_id)

