"""Decode Brotli JSON requests before FastAPI validation."""
import brotli
from fastapi import HTTPException, Request
from fastapi.routing import APIRoute

MAX_BODY_BYTES = 32 * 1024 * 1024


class BrotliRequest(Request):
    async def body(self):
        if hasattr(self, "_body"):
            return self._body
        encodings = self.headers.getlist("content-encoding")
        encoding = ",".join(encodings).strip().lower()
        if encoding in ("", "identity"):
            return await super().body()
        if encoding != "br":
            raise HTTPException(415, "Supported Content-Encoding values: br, identity")
        decoder = brotli.Decompressor()
        output = bytearray()
        received = 0
        try:
            async for chunk in self.stream():
                received += len(chunk)
                if received > MAX_BODY_BYTES:
                    raise HTTPException(413, "Compressed request exceeds 32 MiB")
                # Bound allocation as well as the final decoded body size.
                while True:
                    output.extend(decoder.process(chunk, output_buffer_limit=MAX_BODY_BYTES - len(output) + 1))
                    if len(output) > MAX_BODY_BYTES:
                        raise HTTPException(413, "Decoded request exceeds 32 MiB")
                    if decoder.can_accept_more_data():
                        break
                    chunk = b""
            if not decoder.is_finished():
                raise HTTPException(400, "Incomplete Brotli request body")
        except brotli.error:
            raise HTTPException(400, "Invalid Brotli request body") from None
        self._body = bytes(output)
        return self._body


class BrotliRoute(APIRoute):
    def get_route_handler(self):
        handler = super().get_route_handler()

        async def handle(request: Request):
            return await handler(BrotliRequest(request.scope, request.receive))

        return handle
