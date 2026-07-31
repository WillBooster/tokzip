import { typescriptModule } from '../generated/typescript.ts';
import { registerLanguageModule } from '../moduleRegistry.ts';

// Self-registers on import; the model is validated at registration.
registerLanguageModule(typescriptModule);

export { typescriptModule };
