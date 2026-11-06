import { NextRequest, NextResponse } from "next/server";
import { AccessToken } from "livekit-server-sdk";

export const dynamic = "force-dynamic";

export async function GET(request: NextRequest) {
  try {
    const searchParams = request.nextUrl.searchParams;
    const room = searchParams.get("room") || "voice-room-1";
    const username = searchParams.get("username") || `user-${Math.floor(Math.random() * 10000)}`;

    const apiKey = process.env.LIVEKIT_API_KEY || "devkey";
    const apiSecret = process.env.LIVEKIT_API_SECRET || "secret";
    const wsUrl = process.env.NEXT_PUBLIC_LIVEKIT_URL || "ws://localhost:7880";

    const at = new AccessToken(apiKey, apiSecret, {
      identity: username,
      ttl: "1h",
    });

    at.addGrant({
      room,
      roomJoin: true,
      canPublish: true,
      canSubscribe: true,
    });

    const token = await at.toJwt();

    return NextResponse.json({
      token,
      url: wsUrl,
    });
  } catch (error) {
    console.error("Error generating LiveKit token:", error);
    return NextResponse.json(
      { error: "Failed to create connection token" },
      { status: 500 }
    );
  }
}

