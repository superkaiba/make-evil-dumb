export default function GraphPage() {
  return (
    <div className="flex h-full items-center justify-center text-sm text-neutral-500">
      <div className="max-w-md text-center">
        <h1 className="mb-2 text-lg font-semibold text-neutral-800 dark:text-neutral-200">
          Claim graph
        </h1>
        <p>
          React Flow canvas lands in milestone 3. Nodes will be claims (color by confidence),
          edges by relation type (parent / sibling / supports / contradicts / derives_from).
        </p>
      </div>
    </div>
  );
}
