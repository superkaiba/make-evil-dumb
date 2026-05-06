CREATE TYPE "public"."claim_status" AS ENUM('draft', 'finalized', 'retracted');--> statement-breakpoint
CREATE TYPE "public"."confidence" AS ENUM('HIGH', 'MODERATE', 'LOW');--> statement-breakpoint
CREATE TYPE "public"."edge_type" AS ENUM('parent', 'child', 'sibling', 'supports', 'contradicts', 'derives_from');--> statement-breakpoint
CREATE TYPE "public"."entity_kind" AS ENUM('claim', 'experiment', 'run', 'todo');--> statement-breakpoint
CREATE TYPE "public"."experiment_status" AS ENUM('proposed', 'planning', 'plan_pending', 'approved', 'implementing', 'code_reviewing', 'running', 'uploading', 'interpreting', 'reviewing', 'awaiting_promotion', 'done_experiment', 'done_impl', 'blocked', 'archived');--> statement-breakpoint
CREATE TYPE "public"."message_role" AS ENUM('user', 'assistant', 'tool');--> statement-breakpoint
CREATE TYPE "public"."todo_status" AS ENUM('open', 'in_progress', 'done', 'cancelled');--> statement-breakpoint
CREATE TABLE IF NOT EXISTS "agent_task" (
	"id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"agent" text NOT NULL,
	"stage" text NOT NULL,
	"started_at" timestamp with time zone DEFAULT now() NOT NULL,
	"heartbeat_at" timestamp with time zone DEFAULT now() NOT NULL,
	"completed_at" timestamp with time zone,
	"entity_kind" "entity_kind" NOT NULL,
	"entity_id" uuid NOT NULL
);
--> statement-breakpoint
CREATE TABLE IF NOT EXISTS "chat_message" (
	"id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"session_id" uuid NOT NULL,
	"role" "message_role" NOT NULL,
	"body" text NOT NULL,
	"tool_call_json" jsonb,
	"created_at" timestamp with time zone DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE TABLE IF NOT EXISTS "chat_session" (
	"id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"scope_entity_kind" "entity_kind",
	"scope_entity_id" uuid,
	"agent_handle" text,
	"title" text,
	"created_at" timestamp with time zone DEFAULT now() NOT NULL,
	"last_active_at" timestamp with time zone DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE TABLE IF NOT EXISTS "claim" (
	"id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"title" text NOT NULL,
	"confidence" "confidence",
	"status" "claim_status" DEFAULT 'draft' NOT NULL,
	"body_json" jsonb,
	"hero_figure_id" uuid,
	"github_issue_number" integer,
	"created_at" timestamp with time zone DEFAULT now() NOT NULL,
	"updated_at" timestamp with time zone DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE TABLE IF NOT EXISTS "comment" (
	"id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"author_kind" text NOT NULL,
	"author" text NOT NULL,
	"body" text NOT NULL,
	"entity_kind" "entity_kind" NOT NULL,
	"entity_id" uuid NOT NULL,
	"created_at" timestamp with time zone DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE TABLE IF NOT EXISTS "edge" (
	"from_kind" "entity_kind" NOT NULL,
	"from_id" uuid NOT NULL,
	"to_kind" "entity_kind" NOT NULL,
	"to_id" uuid NOT NULL,
	"type" "edge_type" NOT NULL,
	"created_at" timestamp with time zone DEFAULT now() NOT NULL,
	CONSTRAINT "edge_from_kind_from_id_to_kind_to_id_type_pk" PRIMARY KEY("from_kind","from_id","to_kind","to_id","type")
);
--> statement-breakpoint
CREATE TABLE IF NOT EXISTS "experiment" (
	"id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"title" text NOT NULL,
	"hypothesis" text,
	"plan_json" jsonb,
	"status" "experiment_status" DEFAULT 'proposed' NOT NULL,
	"pod_name" text,
	"parent_id" uuid,
	"claim_id" uuid,
	"github_issue_number" integer,
	"created_at" timestamp with time zone DEFAULT now() NOT NULL,
	"updated_at" timestamp with time zone DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE TABLE IF NOT EXISTS "figure" (
	"id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"url" text NOT NULL,
	"caption" text,
	"entity_kind" "entity_kind" NOT NULL,
	"entity_id" uuid NOT NULL,
	"created_at" timestamp with time zone DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE TABLE IF NOT EXISTS "run" (
	"id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"experiment_id" uuid NOT NULL,
	"seed" integer,
	"config_yaml" text,
	"wandb_url" text,
	"hf_url" text,
	"metrics_json" jsonb,
	"started_at" timestamp with time zone,
	"completed_at" timestamp with time zone,
	"created_at" timestamp with time zone DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE TABLE IF NOT EXISTS "todo" (
	"id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"text" text NOT NULL,
	"due" timestamp with time zone,
	"status" "todo_status" DEFAULT 'open' NOT NULL,
	"linked_kind" "entity_kind",
	"linked_id" uuid,
	"created_at" timestamp with time zone DEFAULT now() NOT NULL
);
--> statement-breakpoint
DO $$ BEGIN
 ALTER TABLE "chat_message" ADD CONSTRAINT "chat_message_session_id_chat_session_id_fk" FOREIGN KEY ("session_id") REFERENCES "public"."chat_session"("id") ON DELETE cascade ON UPDATE no action;
EXCEPTION
 WHEN duplicate_object THEN null;
END $$;
--> statement-breakpoint
DO $$ BEGIN
 ALTER TABLE "experiment" ADD CONSTRAINT "experiment_claim_id_claim_id_fk" FOREIGN KEY ("claim_id") REFERENCES "public"."claim"("id") ON DELETE set null ON UPDATE no action;
EXCEPTION
 WHEN duplicate_object THEN null;
END $$;
--> statement-breakpoint
DO $$ BEGIN
 ALTER TABLE "run" ADD CONSTRAINT "run_experiment_id_experiment_id_fk" FOREIGN KEY ("experiment_id") REFERENCES "public"."experiment"("id") ON DELETE cascade ON UPDATE no action;
EXCEPTION
 WHEN duplicate_object THEN null;
END $$;
--> statement-breakpoint
CREATE INDEX IF NOT EXISTS "agent_task_live_idx" ON "agent_task" USING btree ("completed_at","heartbeat_at");--> statement-breakpoint
CREATE INDEX IF NOT EXISTS "chat_message_session_idx" ON "chat_message" USING btree ("session_id","created_at");--> statement-breakpoint
CREATE INDEX IF NOT EXISTS "comment_entity_idx" ON "comment" USING btree ("entity_kind","entity_id","created_at");--> statement-breakpoint
CREATE INDEX IF NOT EXISTS "edge_from_idx" ON "edge" USING btree ("from_kind","from_id");--> statement-breakpoint
CREATE INDEX IF NOT EXISTS "edge_to_idx" ON "edge" USING btree ("to_kind","to_id");