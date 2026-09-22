import { AnalysisConsole } from "@/components/analysis/analysis-console";
import { readCurrentDatasetSnapshot } from "@/lib/server/csv-adapter";

export const dynamic = "force-dynamic";

export default async function Home() {
  const initialSnapshot = await readCurrentDatasetSnapshot();
  return <AnalysisConsole initialSnapshot={initialSnapshot} />;
}
