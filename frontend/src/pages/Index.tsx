import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Send, GraduationCap, RotateCcw } from "lucide-react";

const QUESTIONS = [
  "Explique o conceito de polimorfismo na programação orientada a objetos.",
  "Qual a diferença entre uma pilha e uma fila em estruturas de dados?",
  "Descreva o funcionamento do protocolo TCP/IP.",
  "O que é normalização de banco de dados? Cite as formas normais.",
  "Explique o princípio da conservação de energia.",
  "Qual a importância do cálculo diferencial na engenharia?",
  "Descreva a Lei de Ohm e suas aplicações práticas.",
  "O que são Design Patterns? Dê exemplos.",
];

const getRandomQuestion = () =>
  QUESTIONS[Math.floor(Math.random() * QUESTIONS.length)];

const getRandomGrade = () => Math.floor(Math.random() * 11);

const GradeDisplay = ({ grade }: { grade: number }) => {
  const color =
    grade >= 6
      ? "text-success"
      : grade >= 4
      ? "text-yellow-500"
      : "text-destructive";

  return (
    <div className="flex flex-col items-center gap-2 animate-fade-in">
      <span className="text-sm font-medium text-muted-foreground uppercase tracking-wider">
        Sua Nota
      </span>
      <span className={`text-8xl font-bold ${color}`}>{grade}</span>
      <span className="text-muted-foreground text-sm">/10</span>
    </div>
  );
};

const Index = () => {
  const [question] = useState(getRandomQuestion);
  const [answer, setAnswer] = useState("");
  const [grade, setGrade] = useState<number | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleSubmit = () => {
    if (!answer.trim()) return;
    setIsSubmitting(true);
    setTimeout(() => {
      setGrade(getRandomGrade());
      setIsSubmitting(false);
    }, 1200);
  };

  const handleReset = () => {
    window.location.reload();
  };

  return (
    <div className="min-h-screen flex flex-col">
      {/* Header */}
      <header className="bg-primary py-4 px-6 shadow-md">
        <div className="max-w-4xl mx-auto flex items-center gap-3">
          <GraduationCap className="h-8 w-8 text-primary-foreground" />
          <h1 className="text-xl font-bold text-primary-foreground tracking-wide">
            Portal Acadêmico
          </h1>
        </div>
      </header>

      {/* Blue accent bar */}
      <div className="h-1 bg-secondary" />

      {/* Main */}
      <main className="flex-1 flex items-center justify-center p-6">
        <div className="w-full max-w-2xl">
          <div className="bg-card rounded-lg shadow-lg overflow-hidden border-t-4 border-t-secondary">
            {/* Card header */}
            <div className="bg-primary px-6 py-4">
              <h2 className="text-primary-foreground font-semibold text-lg">
                Avaliação
              </h2>
            </div>

            <div className="p-6 space-y-6">
              {/* Question */}
              <div className="space-y-2">
                <span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                  Questão
                </span>
                <p className="text-foreground text-lg font-medium leading-relaxed">
                  {question}
                </p>
              </div>

              {grade === null ? (
                <>
                  {/* Answer area */}
                  <div className="space-y-2">
                    <span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                      Sua Resposta
                    </span>
                    <Textarea
                      value={answer}
                      onChange={(e) => setAnswer(e.target.value)}
                      placeholder="Digite sua resposta aqui..."
                      className="min-h-[150px] resize-none text-base focus-visible:ring-secondary"
                      disabled={isSubmitting}
                    />
                  </div>

                  {/* Submit */}
                  <div className="flex justify-end">
                    <Button
                      onClick={handleSubmit}
                      disabled={!answer.trim() || isSubmitting}
                      className="bg-secondary text-secondary-foreground hover:bg-secondary/90 gap-2 px-6"
                    >
                      {isSubmitting ? (
                        <div className="h-4 w-4 border-2 border-secondary-foreground/30 border-t-secondary-foreground rounded-full animate-spin" />
                      ) : (
                        <Send className="h-4 w-4" />
                      )}
                      {isSubmitting ? "Corrigindo..." : "Enviar Resposta"}
                    </Button>
                  </div>
                </>
              ) : (
                /* Grade result */
                <div className="flex flex-col items-center gap-6 py-8">
                  <GradeDisplay grade={grade} />
                  <Button
                    onClick={handleReset}
                    variant="outline"
                    className="gap-2 border-secondary text-secondary hover:bg-secondary/10"
                  >
                    <RotateCcw className="h-4 w-4" />
                    Nova Questão
                  </Button>
                </div>
              )}
            </div>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="bg-primary py-3 text-center">
        <p className="text-primary-foreground/60 text-sm">
          Portal Acadêmico © 2026
        </p>
      </footer>
    </div>
  );
};

export default Index;
