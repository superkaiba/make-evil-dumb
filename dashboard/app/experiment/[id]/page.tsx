export default async function ExperimentPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  return (
    <div className="p-6">
      <h1 className="mb-2 text-lg font-semibold">Experiment</h1>
      <p className="text-sm text-neutral-500">id: {id}</p>
    </div>
  );
}
