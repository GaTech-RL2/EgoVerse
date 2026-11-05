import { NextResponse } from "next/server";

import { getAnalysisJob } from "@/lib/server/job-store";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET(
  _request: Request,
  context: { params: Promise<{ jobId: string }> },
) {
  const { jobId } = await context.params;
  try {
    return NextResponse.json(await getAnalysisJob(jobId));
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Job not found." },
      { status: 404 },
    );
  }
}
