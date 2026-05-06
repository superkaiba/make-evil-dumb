import { drizzle, type PostgresJsDatabase } from "drizzle-orm/postgres-js";
import postgres from "postgres";
import * as schema from "./schema";

declare global {
  // eslint-disable-next-line no-var
  var __pg: ReturnType<typeof postgres> | undefined;
  // eslint-disable-next-line no-var
  var __db: PostgresJsDatabase<typeof schema> | undefined;
}

function init(): PostgresJsDatabase<typeof schema> {
  const url = process.env.DATABASE_URL;
  if (!url) {
    throw new Error("DATABASE_URL is not set. Add it to dashboard/.env.local.");
  }
  const client = global.__pg ?? postgres(url, { max: 10, prepare: false });
  if (process.env.NODE_ENV !== "production") global.__pg = client;
  return drizzle(client, { schema });
}

export function getDb(): PostgresJsDatabase<typeof schema> {
  if (!global.__db) global.__db = init();
  return global.__db;
}

export { schema };
