import { NextResponse } from "next/server";

import { getAnalysisResult } from "@/lib/server/job-store";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET(
  _request: Request,
  context: { params: Promise<{ jobId: string }> },
) {
  const { jobId } = await context.params;
  try {
    return NextResponse.json(await getAnalysisResult(jobId));
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Result not found." },
      { status: 409 },
    );
  }
}
