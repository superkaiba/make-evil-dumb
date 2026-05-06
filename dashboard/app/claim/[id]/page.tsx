export default async function ClaimPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  return (
    <div className="p-6">
      <h1 className="mb-2 text-lg font-semibold">Claim</h1>
      <p className="text-sm text-neutral-500">id: {id}</p>
      <p className="mt-4 text-sm text-neutral-500">
        Detail page (hero figure, TipTap body, linked rail, comments) lands in milestone 4.
      </p>
    </div>
  );
}
