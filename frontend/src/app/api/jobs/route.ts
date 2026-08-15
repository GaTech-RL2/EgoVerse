import { NextResponse } from "next/server";

import {
  ActiveJobError,
  BackendEnvironmentError,
  createAnalysisJob,
} from "@/lib/server/job-store";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function POST() {
  try {
    const job = await createAnalysisJob();
    return NextResponse.json(job, { status: 202 });
  } catch (error) {
    if (error instanceof ActiveJobError) {
      return NextResponse.json(
        {
          error: error.message,
          activeJobId: error.activeJobId,
        },
        { status: 409 },
      );
    }
    if (error instanceof BackendEnvironmentError) {
      return NextResponse.json({ error: error.message }, { status: 503 });
    }
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Unable to create analysis job." },
      { status: 500 },
    );
  }
}
