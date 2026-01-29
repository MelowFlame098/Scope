package worker

import (
	"encoding/json"
	"fmt"
	"log"

	"github.com/hibiken/asynq"
	"github.com/redis/go-redis/v9"
)

// Task Types
const (
	TypeEmailDelivery    = "email:deliver"
	TypePortfolioUpdate  = "portfolio:update"
	TypeDataIngestion    = "data:ingest"
	TypeAutomatedTrading = "trading:automated"
)

// Task Payloads
type EmailDeliveryPayload struct {
	UserID  string
	Subject string
	Body    string
}

type PortfolioUpdatePayload struct {
	PortfolioID string
}

// TaskDistributorInterface interface
type TaskDistributorInterface interface {
	DistributeTaskSendEmail(payload *EmailDeliveryPayload, opts ...asynq.Option) error
	DistributeTaskPortfolioUpdate(payload *PortfolioUpdatePayload, opts ...asynq.Option) error
	DistributeTaskAutomatedTrading(opts ...asynq.Option) error
}

type TaskDistributor struct {
	client *asynq.Client
}

func NewTaskDistributor(rdb *redis.Client) *TaskDistributor {
	// Re-using the same connection params for Asynq
	// In production, might want separate config
	client := asynq.NewClient(asynq.RedisClientOpt{Addr: rdb.Options().Addr})
	return &TaskDistributor{client: client}
}

func (distributor *TaskDistributor) DistributeTaskSendEmail(payload *EmailDeliveryPayload, opts ...asynq.Option) error {
	jsonPayload, err := json.Marshal(payload)
	if err != nil {
		return fmt.Errorf("failed to marshal task payload: %w", err)
	}
	task := asynq.NewTask(TypeEmailDelivery, jsonPayload, opts...)
	info, err := distributor.client.Enqueue(task)
	if err != nil {
		return fmt.Errorf("failed to enqueue task: %w", err)
	}
	log.Printf("Enqueued task: id=%s queue=%s", info.ID, info.Queue)
	return nil
}

func (distributor *TaskDistributor) DistributeTaskPortfolioUpdate(payload *PortfolioUpdatePayload, opts ...asynq.Option) error {
	jsonPayload, err := json.Marshal(payload)
	if err != nil {
		return fmt.Errorf("failed to marshal task payload: %w", err)
	}
	task := asynq.NewTask(TypePortfolioUpdate, jsonPayload, opts...)
	info, err := distributor.client.Enqueue(task)
	if err != nil {
		return fmt.Errorf("failed to enqueue task: %w", err)
	}
	log.Printf("Enqueued task: id=%s queue=%s", info.ID, info.Queue)
	return nil
}

func (distributor *TaskDistributor) DistributeTaskAutomatedTrading(opts ...asynq.Option) error {
	task := asynq.NewTask(TypeAutomatedTrading, nil, opts...)
	info, err := distributor.client.Enqueue(task)
	if err != nil {
		return fmt.Errorf("failed to enqueue task: %w", err)
	}
	log.Printf("Enqueued automated trading task: id=%s queue=%s", info.ID, info.Queue)
	return nil
}
