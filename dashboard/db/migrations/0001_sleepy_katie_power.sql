ALTER TABLE "todo" ADD COLUMN "github_issue_number" integer;--> statement-breakpoint
ALTER TABLE "claim" ADD CONSTRAINT "claim_github_issue_number_unique" UNIQUE("github_issue_number");--> statement-breakpoint
ALTER TABLE "experiment" ADD CONSTRAINT "experiment_github_issue_number_unique" UNIQUE("github_issue_number");--> statement-breakpoint
ALTER TABLE "todo" ADD CONSTRAINT "todo_github_issue_number_unique" UNIQUE("github_issue_number");