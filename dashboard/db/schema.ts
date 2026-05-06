import {
  pgTable,
  pgEnum,
  uuid,
  text,
  integer,
  timestamp,
  jsonb,
  primaryKey,
  index,
} from "drizzle-orm/pg-core";

export const confidenceEnum = pgEnum("confidence", ["HIGH", "MODERATE", "LOW"]);

export const claimStatusEnum = pgEnum("claim_status", [
  "draft",
  "finalized",
  "retracted",
]);

export const experimentStatusEnum = pgEnum("experiment_status", [
  "proposed",
  "planning",
  "plan_pending",
  "approved",
  "implementing",
  "code_reviewing",
  "running",
  "uploading",
  "interpreting",
  "reviewing",
  "awaiting_promotion",
  "done_experiment",
  "done_impl",
  "blocked",
  "archived",
]);

export const edgeTypeEnum = pgEnum("edge_type", [
  "parent",
  "child",
  "sibling",
  "supports",
  "contradicts",
  "derives_from",
]);

export const entityKindEnum = pgEnum("entity_kind", [
  "claim",
  "experiment",
  "run",
  "todo",
]);

export const todoStatusEnum = pgEnum("todo_status", [
  "open",
  "in_progress",
  "done",
  "cancelled",
]);

export const messageRoleEnum = pgEnum("message_role", [
  "user",
  "assistant",
  "tool",
]);

export const claims = pgTable("claim", {
  id: uuid("id").primaryKey().defaultRandom(),
  title: text("title").notNull(),
  confidence: confidenceEnum("confidence"),
  status: claimStatusEnum("status").notNull().default("draft"),
  bodyJson: jsonb("body_json"),
  heroFigureId: uuid("hero_figure_id"),
  githubIssueNumber: integer("github_issue_number"),
  createdAt: timestamp("created_at", { withTimezone: true }).defaultNow().notNull(),
  updatedAt: timestamp("updated_at", { withTimezone: true }).defaultNow().notNull(),
});

export const experiments = pgTable("experiment", {
  id: uuid("id").primaryKey().defaultRandom(),
  title: text("title").notNull(),
  hypothesis: text("hypothesis"),
  planJson: jsonb("plan_json"),
  status: experimentStatusEnum("status").notNull().default("proposed"),
  podName: text("pod_name"),
  parentId: uuid("parent_id"),
  claimId: uuid("claim_id").references(() => claims.id, { onDelete: "set null" }),
  githubIssueNumber: integer("github_issue_number"),
  createdAt: timestamp("created_at", { withTimezone: true }).defaultNow().notNull(),
  updatedAt: timestamp("updated_at", { withTimezone: true }).defaultNow().notNull(),
});

export const runs = pgTable("run", {
  id: uuid("id").primaryKey().defaultRandom(),
  experimentId: uuid("experiment_id")
    .references(() => experiments.id, { onDelete: "cascade" })
    .notNull(),
  seed: integer("seed"),
  configYaml: text("config_yaml"),
  wandbUrl: text("wandb_url"),
  hfUrl: text("hf_url"),
  metricsJson: jsonb("metrics_json"),
  startedAt: timestamp("started_at", { withTimezone: true }),
  completedAt: timestamp("completed_at", { withTimezone: true }),
  createdAt: timestamp("created_at", { withTimezone: true }).defaultNow().notNull(),
});

export const todos = pgTable("todo", {
  id: uuid("id").primaryKey().defaultRandom(),
  text: text("text").notNull(),
  due: timestamp("due", { withTimezone: true }),
  status: todoStatusEnum("status").notNull().default("open"),
  linkedKind: entityKindEnum("linked_kind"),
  linkedId: uuid("linked_id"),
  createdAt: timestamp("created_at", { withTimezone: true }).defaultNow().notNull(),
});

export const edges = pgTable(
  "edge",
  {
    fromKind: entityKindEnum("from_kind").notNull(),
    fromId: uuid("from_id").notNull(),
    toKind: entityKindEnum("to_kind").notNull(),
    toId: uuid("to_id").notNull(),
    type: edgeTypeEnum("type").notNull(),
    createdAt: timestamp("created_at", { withTimezone: true }).defaultNow().notNull(),
  },
  (t) => ({
    pk: primaryKey({ columns: [t.fromKind, t.fromId, t.toKind, t.toId, t.type] }),
    fromIdx: index("edge_from_idx").on(t.fromKind, t.fromId),
    toIdx: index("edge_to_idx").on(t.toKind, t.toId),
  }),
);

export const figures = pgTable("figure", {
  id: uuid("id").primaryKey().defaultRandom(),
  url: text("url").notNull(),
  caption: text("caption"),
  entityKind: entityKindEnum("entity_kind").notNull(),
  entityId: uuid("entity_id").notNull(),
  createdAt: timestamp("created_at", { withTimezone: true }).defaultNow().notNull(),
});

export const comments = pgTable(
  "comment",
  {
    id: uuid("id").primaryKey().defaultRandom(),
    authorKind: text("author_kind").notNull(),
    author: text("author").notNull(),
    body: text("body").notNull(),
    entityKind: entityKindEnum("entity_kind").notNull(),
    entityId: uuid("entity_id").notNull(),
    createdAt: timestamp("created_at", { withTimezone: true }).defaultNow().notNull(),
  },
  (t) => ({
    entityIdx: index("comment_entity_idx").on(t.entityKind, t.entityId, t.createdAt),
  }),
);

export const agentTasks = pgTable(
  "agent_task",
  {
    id: uuid("id").primaryKey().defaultRandom(),
    agent: text("agent").notNull(),
    stage: text("stage").notNull(),
    startedAt: timestamp("started_at", { withTimezone: true }).defaultNow().notNull(),
    heartbeatAt: timestamp("heartbeat_at", { withTimezone: true }).defaultNow().notNull(),
    completedAt: timestamp("completed_at", { withTimezone: true }),
    entityKind: entityKindEnum("entity_kind").notNull(),
    entityId: uuid("entity_id").notNull(),
  },
  (t) => ({
    liveIdx: index("agent_task_live_idx").on(t.completedAt, t.heartbeatAt),
  }),
);

export const chatSessions = pgTable("chat_session", {
  id: uuid("id").primaryKey().defaultRandom(),
  scopeEntityKind: entityKindEnum("scope_entity_kind"),
  scopeEntityId: uuid("scope_entity_id"),
  agentHandle: text("agent_handle"),
  title: text("title"),
  createdAt: timestamp("created_at", { withTimezone: true }).defaultNow().notNull(),
  lastActiveAt: timestamp("last_active_at", { withTimezone: true }).defaultNow().notNull(),
});

export const chatMessages = pgTable(
  "chat_message",
  {
    id: uuid("id").primaryKey().defaultRandom(),
    sessionId: uuid("session_id")
      .references(() => chatSessions.id, { onDelete: "cascade" })
      .notNull(),
    role: messageRoleEnum("role").notNull(),
    body: text("body").notNull(),
    toolCallJson: jsonb("tool_call_json"),
    createdAt: timestamp("created_at", { withTimezone: true }).defaultNow().notNull(),
  },
  (t) => ({
    sessionIdx: index("chat_message_session_idx").on(t.sessionId, t.createdAt),
  }),
);

export type Claim = typeof claims.$inferSelect;
export type NewClaim = typeof claims.$inferInsert;
export type Experiment = typeof experiments.$inferSelect;
export type NewExperiment = typeof experiments.$inferInsert;
export type Run = typeof runs.$inferSelect;
export type NewRun = typeof runs.$inferInsert;
export type Todo = typeof todos.$inferSelect;
export type Edge = typeof edges.$inferSelect;
export type Figure = typeof figures.$inferSelect;
export type Comment = typeof comments.$inferSelect;
export type AgentTask = typeof agentTasks.$inferSelect;
export type ChatSession = typeof chatSessions.$inferSelect;
export type ChatMessage = typeof chatMessages.$inferSelect;
