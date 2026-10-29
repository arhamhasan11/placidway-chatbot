# Write-up

## Known limitations

- **Only the 7 pages.** Anything else gets an honest "not on these pages" and a quote link.
- **Prices are as of the last refresh, in USD.** The bot never converts currencies or calculates totals.
- **Comparisons are limited by the content.** The comparison page has no cancer prices, so cross-country comparisons only work for the alternative-medicine clinics it lists.
- **The videos page gives titles only.** There are no transcripts.
- **The price check catches invented numbers, not misplaced ones.** A real price attached to the wrong item passes the code check; the prompt rules and the eval set guard against this.
- **Answers vary between runs.** A detail (an add-on price, a qualifier) is sometimes left out; repeated full evaluations score 34–36 of 38, with all 8 brief tests passing.
- **I wrote the 38 evaluation questions myself**, so they may share my blind spots.

## With another week

1. **Check every sentence, not just prices,** against the source it cites, and measure how often that wrongly rejects good answers.
2. **Price-change alerts:** each refresh already produces a diff of the pages. Notify PlacidWay when a price changes or disappears.
3. **Weekly content-gap report** from the logged unanswered questions, so the content team knows what to write next.
4. **Grow the eval set from real visitor questions** and block prompt or model changes that lower the score.
5. **Transcribe the 128 ITC videos**, the largest content the bot can't read yet.
6. **Send leads to PlacidWay's CRM** with the chat transcript, and hand off to a human on request.
