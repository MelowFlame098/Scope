package worker

import (
	"context"
	"encoding/json"
	"fmt"
	"log"

	"github.com/hibiken/asynq"
	"github.com/redis/go-redis/v9"
)

func StartWorkerServer(rdb *redis.Client) {
	srv := asynq.NewServer(
		asynq.RedisClientOpt{Addr: rdb.Options().Addr},
		asynq.Config{
			Concurrency: 10,
			Queues: map[string]int{
				"critical": 6,
				"default":  3,
				"low":      1,
			},
		},
	)

	mux := asynq.NewServeMux()
	mux.HandleFunc(TypeEmailDelivery, HandleEmailDeliveryTask)
	mux.HandleFunc(TypePortfolioUpdate, HandlePortfolioUpdateTask)

	if err := srv.Run(mux); err != nil {
		log.Fatalf("could not run server: %v", err)
	}
}

func HandleEmailDeliveryTask(ctx context.Context, task *asynq.Task) error {
	var payload EmailDeliveryPayload
	if err := json.Unmarshal(task.Payload(), &payload); err != nil {
		return fmt.Errorf("json.Unmarshal failed: %v: %w", err, asynq.SkipRetry)
	}
	log.Printf("Sending Email to User: user_id=%s, template=%s", payload.UserID, payload.Subject)
	// Email sending logic...
	return nil
}

func HandlePortfolioUpdateTask(ctx context.Context, task *asynq.Task) error {
	var payload PortfolioUpdatePayload
	if err := json.Unmarshal(task.Payload(), &payload); err != nil {
		return fmt.Errorf("json.Unmarshal failed: %v: %w", err, asynq.SkipRetry)
	}
	log.Printf("Updating Portfolio: portfolio_id=%s", payload.PortfolioID)
	// Portfolio update logic...
	return nil
}
