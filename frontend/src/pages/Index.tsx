import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Send, RotateCcw } from "lucide-react";

const QUESTIONS = [
  "Qual o significado da vida?",
  "Explique computação quântica em termos simples.",
  "O que torna um bom líder?",
  "Porque o céu é azul?",
  "Qual é a invenção mais importante de todos os tempos?",
  "Como você salvaria a fome humana?",
  "O que é consciência?",
  "Descreva internet para um cavaleiro medieval.",
  "O que vem depois do infinito?",
  "Porque sonhamos?",
];

function getRandomQuestion() {
  return QUESTIONS[Math.floor(Math.random() * QUESTIONS.length)];
}

function getGradeColor(grade: number) {
  if (grade >= 9) return "text-grade-excellent";
  if (grade >= 7) return "text-grade-good";
  if (grade >= 5) return "text-grade-ok";
  if (grade >= 3) return "text-grade-bad";
  return "text-grade-fail";
}

function getGradeLabel(grade: number) {
  if (grade === 10) return "Perfeito!";
  if (grade >= 9) return "Excelente!";
  if (grade >= 7) return "Bom trabalho!";
  if (grade >= 5) return "Não está ruim";
  if (grade >= 3) return "Precisa melhorar";
  return "Putz...";
}

const Index = () => {
  const [question] = useState(getRandomQuestion);
  const [answer, setAnswer] = useState("");
  const [grade, setGrade] = useState<number | null>(null);
  const [isGrading, setIsGrading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!answer.trim() || isGrading) return;

    setIsGrading(true);
    
    try {
      const response = await fetch(`/api/v1/answer/5`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ 
          resposta: answer,
          pergunta: question 
        }),
      });

      if (!response.ok) throw new Error("Failed to grade");
      
      const data = await response.json();
      console.log("API Response:", data);
      
      const score = data.data?.score || 0;
      console.log("Extracted Score:", score);
      setGrade(Math.min(Math.max(Math.round(score), 0), 10));
      
    } catch (error) {
      console.error("Grading error:", error);
      alert("Erro ao avaliar a resposta. Verifique se a API está rodando em 127.0.0.1:3000");
      setGrade(0);
    } finally {
      setIsGrading(false);
    }
  };

  const handleReset = () => {
    setAnswer("");
    setGrade(null);
    // Force remount with new question
    window.location.reload();
  };

  return (
    <div className="flex min-h-screen flex-col items-center justify-center bg-background px-4">
      <motion.div
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, ease: "easeOut" }}
        className="w-full max-w-xl"
      >
        {/* Header */}
        <div className="mb-10 text-center">
          <h1 className="font-mono text-sm font-semibold uppercase tracking-widest text-muted-foreground">
            QuestIA
          </h1>
        </div>

        {/* Question Card */}
        <div className="rounded-2xl border border-border bg-card p-8 shadow-sm">
          <p className="mb-1 font-mono text-xs font-medium uppercase tracking-wider text-muted-foreground">
            Pergunta
          </p>
          <h2 className="mb-8 text-2xl font-bold leading-tight text-card-foreground">
            {question}
          </h2>

          <AnimatePresence mode="wait">
            {grade === null ? (
              <motion.form
                key="form"
                onSubmit={handleSubmit}
                initial={{ opacity: 1 }}
                exit={{ opacity: 0, y: -20 }}
                transition={{ duration: 0.3 }}
              >
                <textarea
                  value={answer}
                  onChange={(e) => setAnswer(e.target.value)}
                  placeholder="Escreva sua resposta aqui..."
                  rows={4}
                  className="w-full resize-none rounded-xl border border-input bg-background px-4 py-3 text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring"
                />
                <button
                  type="submit"
                  disabled={!answer.trim() || isGrading}
                  className="mt-4 flex w-full items-center justify-center gap-2 rounded-xl bg-primary px-6 py-3 font-semibold text-primary-foreground transition-all hover:opacity-90 disabled:opacity-40 disabled:cursor-not-allowed"
                >
                  {isGrading ? (
                    <motion.div
                      animate={{ rotate: 360 }}
                      transition={{ repeat: Infinity, duration: 0.8, ease: "linear" }}
                    >
                      <Send className="h-5 w-5" />
                    </motion.div>
                  ) : (
                    <Send className="h-5 w-5" />
                  )}
                  {isGrading ? "Avaliando..." : "Enviar Resposta"}
                </button>
              </motion.form>
            ) : (
              <motion.div
                key="result"
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.5, type: "spring", bounce: 0.4 }}
                className="text-center"
              >
                <p className="mb-2 font-mono text-xs font-medium uppercase tracking-wider text-muted-foreground">
                  Sua nota é 
                </p>
                <motion.p
                  initial={{ scale: 0 }}
                  animate={{ scale: 1 }}
                  transition={{ delay: 0.2, type: "spring", bounce: 0.5 }}
                  className={`font-mono text-8xl font-bold ${getGradeColor(grade)}`}
                >
                  {grade}
                </motion.p>
                <p className="mt-1 font-mono text-sm text-muted-foreground">/10</p>
                <p className={`mt-3 text-lg font-semibold ${getGradeColor(grade)}`}>
                  {getGradeLabel(grade)}
                </p>

                <div className="mt-6 rounded-xl border border-border bg-background p-4">
                  <p className="text-sm italic text-muted-foreground">
                    "{answer}"
                  </p>
                </div>

                <button
                  onClick={handleReset}
                  className="mt-6 inline-flex items-center gap-2 rounded-xl border border-border bg-secondary px-6 py-3 font-semibold text-secondary-foreground transition-all hover:opacity-80"
                >
                  <RotateCcw className="h-4 w-4" />
                  Tente Outra
                </button>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </motion.div>
    </div>
  );
};

export default Index;
